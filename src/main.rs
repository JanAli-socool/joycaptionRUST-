use anyhow::{anyhow, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use image::imageops::FilterType;
use image::GenericImageView;
use serde::Deserialize;
use std::{collections::HashSet, fs, path::PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llava::config::{
    HFGenerationConfig, HFLLaVAConfig, HFPreProcessorConfig, LLaVAConfig,
};
use candle_transformers::models::llava::LLaVA;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// HF model id
    #[arg(long, default_value = "fancyfeast/llama-joycaption-beta-one-hf-llava")]
    model_id: String,

    /// Path to image to caption
    #[arg(long)]
    image: String,

    /// Text prompt (e.g., "Describe the image.")
    #[arg(long, default_value = "Describe the image in detail.")]
    prompt: String,

    /// Max new tokens
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (<=0 means greedy)
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Force CPU (fallback is automatic if CUDA not available)
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

// --- constants matching LLaVA formatting ---
const DEFAULT_IMAGE_TOKEN: &str = "<image>";
const DEFAULT_IM_START_TOKEN: &str = "<im_start>";
const DEFAULT_IM_END_TOKEN: &str = "<im_end>";

fn main() -> Result<()> {
    let args = Args::parse();

    // ---------- device ----------
    // Prefer CUDA 0, fallback to CPU silently
    let device = if args.cpu {
        Device::Cpu
    } else {
        match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        }
    };

    // ---------- HF Hub fetch ----------
    let api = Api::new()?;
    let repo = api.model(args.model_id.clone());

    let config_path = repo.get("config.json")?;
    let gen_config_path = repo.get("generation_config.json")?;
    let preproc_path = repo.get("preprocessor_config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?; // present in the hf-llava repo

    // Many JoyCaption weights are sharded + indexed
    let index_path = repo.get("model.safetensors.index.json")?;
    let shard_paths = gather_safetensor_shards(&repo, index_path)?;

    // ---------- parse configs ----------
    let hf_llava_config: HFLLaVAConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    let hf_gen_config: HFGenerationConfig = serde_json::from_slice(&fs::read(gen_config_path)?)?;
    let hf_preproc: HFPreProcessorConfig = serde_json::from_slice(&fs::read(preproc_path)?)?;

    let llava_config = hf_llava_config.to_llava_config(&hf_gen_config, &hf_preproc);
    let clip_vision_config = hf_llava_config.to_clip_vision_config();

    // dtype from config.json (bfloat16 recommended)
    let dtype = match llava_config.torch_dtype.as_str() {
        "bfloat16" => DType::BF16,
        "float16" => DType::F16,
        _ => DType::F32,
    };

    // ---------- load model ----------
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device)? };
    let llava: LLaVA = LLaVA::load(vb, &llava_config, Some(clip_vision_config))?;

    // ---------- tokenizer ----------
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("{e}"))?;

    // ---------- image preprocess (384x384, RGB, [-1,1] normalized) ----------
    // Matches the JoyCaption Python that resizes (squish) to a square 384 with LANCZOS
    let (pixel_values, image_size) = preprocess_joycaption_image(&args.image, 384, &device)?;

    // ---------- build prompt with image token ----------
    let qs = if llava_config.mm_use_im_start_end {
        // <im_start><image><im_end>\nPROMPT
        format!(
            "{}{}{}\n{}",
            DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, args.prompt
        )
    } else {
        format!("{}\n{}", DEFAULT_IMAGE_TOKEN, args.prompt)
    };

    // Insert special image token id at <image> position(s)
    let tokens = tokenizer_image_token(
        &qs,
        &tokenizer,
        llava_config.image_token_index as i64,
        &llava_config,
    )?;

    // Prepare multimodal inputs for LLaVA
    let mut input_embeds =
        llava.prepare_inputs_labels_for_multimodal(&tokens, &[pixel_values], &[image_size])?;

    // ---------- generation loop (simple greedy/sampling) ----------
    let mut cache = candle_transformers::models::llama::Cache::new(
        true,
        dtype,
        &llava_config.to_llama_config(),
        &device,
    )?;

    let mut logits_processor = {
        let t = f64::from(args.temperature);
        let sampling = if t <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature: t }
        };
        candle_transformers::generation::LogitsProcessor::from_sampling(299792458, sampling)
    };

    let eos = llava_config.eos_token_id as u32;
    let mut collected: Vec<u32> = Vec::new();
    let mut index_pos = 0usize;

    for step in 0..args.max_new_tokens {
        let (_, seqlen, _) = input_embeds.dims3()?;
        let (ctx, ctx_index) = if cache.use_kv_cache && step > 0 {
            (1, index_pos)
        } else {
            (seqlen, 0)
        };

        let input = input_embeds.i((.., seqlen.saturating_sub(ctx).., ..))?;
        let logits = llava.forward(&input, ctx_index, &mut cache)?.squeeze(0)?;
        let (_, in_len, _) = input.dims3()?;
        index_pos += in_len;

        let next = logits_processor.sample(&logits)?;
        collected.push(next);

        if next == eos {
            break;
        }

        // next token → embedding
        let next_t = Tensor::from_vec(vec![next], 1, &device)?;
        let next_embeds = llava.llama.embed(&next_t)?.unsqueeze(0)?;
        input_embeds = Tensor::cat(&[input_embeds, next_embeds], 1)?;
    }

    // ---------- decode ----------
    let caption = decode_tokens(&tokenizer, &collected)?;
    println!("{caption}");

    Ok(())
}

// ---- helpers ----

fn gather_safetensor_shards(repo: &hf_hub::api::sync::ApiRepo, index_path: PathBuf) -> Result<Vec<PathBuf>> {
    #[derive(Deserialize)]
    struct IndexFile {
        weight_map: serde_json::Map<String, serde_json::Value>,
    }
    let idx: IndexFile = serde_json::from_slice(&fs::read(index_path)?)?;
    let mut uniq = HashSet::new();
    for v in idx.weight_map.values() {
        if let Some(fname) = v.as_str() {
            uniq.insert(fname.to_string());
        }
    }
    let mut files = Vec::new();
    for fname in uniq {
        files.push(repo.get(&fname)?);
    }
    Ok(files)
}

fn preprocess_joycaption_image(path: &str, size: u32, device: &Device) -> Result<(Tensor, (u32, u32))> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (w, h) = img.dimensions();

    // Squash-resize to exactly size×size, like the JoyCaption Python script
    let img = img.resize_exact(size, size, FilterType::Lanczos3).to_rgb8();

    // To f32 tensor [1,3,H,W] and normalize to [-1, 1] via (x/255 - 0.5) / 0.5
    let mut data_f32 = Vec::with_capacity((size * size * 3) as usize);
    for px in img.pixels() {
        data_f32.push(px[0] as f32 / 255.0);
        data_f32.push(px[1] as f32 / 255.0);
        data_f32.push(px[2] as f32 / 255.0);
    }

    // HWC -> CHW
    // HWC -> CHW
    let hw = (size * size) as usize;
    let inter = data_f32;
    
    // Split into channels properly
    let mut rch = Vec::with_capacity(hw);
    let mut gch = Vec::with_capacity(hw);
    let mut bch = Vec::with_capacity(hw);
    for i in (0..inter.len()).step_by(3) {
        rch.push(inter[i]);
        gch.push(inter[i + 1]);
        bch.push(inter[i + 2]);
    }
    
    // Combine as CHW
    let mut chw = Vec::with_capacity(inter.len());
    chw.extend(rch);
    chw.extend(gch);
    chw.extend(bch);
    
    // Normalize [-1,1]
    for v in chw.iter_mut() {
        *v = (*v - 0.5) / 0.5;
    }
    
    let tensor = Tensor::from_vec(chw, (1, 3, size as usize, size as usize), device)?;
    Ok((tensor, (w, h)))
    // let hw = (size * size) as usize;
    // // let (r, g, b) = (&data_f32[0..hw], &data_f32[hw..2*hw], &data_f32[2*hw..3*hw]); // actually we pushed interleaved; so reorder properly
    // let (r, g, _b): (&[f32], &[f32], &[f32]) = (
    // &data_f32[0..hw],
    // &data_f32[hw..2*hw],
    // &data_f32[2*hw..3*hw],
    // );

    // let mut chw = Vec::with_capacity(data_f32.len());
    // // Rebuild correctly: iterate pixels and push channels per channel
    // chw.clear();
    // // We'll rebuild from the interleaved buffer
    // let inter = data_f32;
    // let mut rch = Vec::with_capacity(hw);
    // let mut gch = Vec::with_capacity(hw);
    // let mut bch = Vec::with_capacity(hw);
    // for i in (0..inter.len()).step_by(3) {
    //     rch.push(inter[i]);
    //     gch.push(inter[i + 1]);
    //     bch.push(inter[i + 2]);
    // }
    // chw.extend(rch);
    // chw.extend(gch);
    // chw.extend(bch);

    // Normalize to [-1,1]
    for v in chw.iter_mut() {
        *v = (*v - 0.5) / 0.5;
    }

    let tensor = Tensor::from_vec(chw, (1, 3, size as usize, size as usize), device)?;
    Ok((tensor, (w, h)));
}

fn tokenizer_image_token(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_index: i64,
    llava_config: &LLaVAConfig,
) -> Result<Tensor> {
    // Split on "<image>" and stitch ids back, inserting the image_token_index
    let chunks: Vec<Vec<i64>> = prompt
        .split(DEFAULT_IMAGE_TOKEN)
        .map(|s| {
            tokenizer
                .encode(s, true)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&x| x as i64)
                .collect()
        })
        .collect();

    let mut input_ids: Vec<i64> = Vec::new();
    let mut offset = 0;

    // Keep BOS if it's present at the very start
    if !chunks.is_empty()
        && !chunks[0].is_empty()
        && chunks[0][0] == llava_config.bos_token_id as i64
    {
        offset = 1;
        input_ids.push(chunks[0][0]);
    }

    // Interleave chunks with image token(s)
    for (i, part) in chunks.iter().enumerate() {
        if i == 0 {
            input_ids.extend(part.iter().skip(offset));
        } else {
            input_ids.push(image_token_index);
            input_ids.extend(part.iter());
        }
    }

    let len = input_ids.len();
    Tensor::from_vec(input_ids, (1, len), &Device::Cpu).map_err(anyhow::Error::msg)
}

fn decode_tokens(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    // tokenizer.decode expects &[u32]
    // let s = tokenizer.decode(ids.to_vec(), true).map_err(|e| anyhow!("{e}"))?;
    let s = tokenizer.decode(ids.as_slice(), true).map_err(|e| anyhow!("{e}"))?;
    Ok(s.trim().to_string())
}

use anyhow::{bail, Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::Cache;
use candle_transformers::models::llava::config::{
    HFGenerationConfig, HFLLaVAConfig, HFPreProcessorConfig, LLaVAConfig,
};
use candle_transformers::models::llava::{LLaVA};
use clap::Parser;
use hf_hub::api::sync::Api;
use image::io::Reader as ImageReader;
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// HF repo or local directory containing safetensors + config.json
    #[arg(long, default_value = "llava-hf/llava-v1.6-vicuna-7b-hf")]
    model_path: String,

    /// Path to tokenizer.json (ignored if --hf is set and tokenizer in repo)
    #[arg(long, default_value = "tokenizer/tokenizer.json")]
    tokenizer_path: String,

    /// Input image file (required)
    #[arg(long)]
    image_file: String,

    /// Prompt to condition generation
    #[arg(long, default_value = "Describe the image in detail.")]
    prompt: String,

    /// Use HuggingFace repo layout (config/generation_config/preprocessor/tokenizer)
    #[arg(long, action)]
    hf: bool,

    /// Force CPU (default). Use --gpu to request CUDA if available.
    #[arg(long, action)]
    gpu: bool,

    /// Disable KV cache
    #[arg(long, action)]
    no_kv_cache: bool,

    /// Max new tokens to generate
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (<=0 means greedy)
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// RNG seed for sampling
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

fn device_from_args(gpu: bool) -> Result<Device> {
    if gpu {
        Device::new_cuda(0).or_else(|_| {
            eprintln!("CUDA not available, falling back to CPU");
            Ok(Device::Cpu)
        })
    } else {
        Ok(Device::Cpu)
    }
}

fn load_image_tensor(
    path: &str,
    processor: &candle_examples::image_proc::ImageProcessor, // helper from candle examples
    llava_cfg: &LLaVAConfig,
    dtype: DType,
) -> Result<((u32, u32), Tensor)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("failed to open image: {}", path))?
        .decode()
        .context("failed to decode image")?;
    let (w, h) = (img.width(), img.height());
    let t = processor
        .process_image(&img, llava_cfg)
        .context("image preprocessing failed")?
        .to_dtype(dtype)?;
    Ok(((w, h), t))
}

/// Tokenize with support for <image> placeholder expansion per LLaVA config.
fn tokenize_with_image_tokens(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_index: i64,
    llava_cfg: &LLaVAConfig,
) -> Result<Tensor> {
    // Split prompt by the image placeholder used in HF LLaVA flows.
    let chunks = prompt
        .split("<image>")
        .map(|s| {
            tokenizer
                .encode(s, true)
                .map(|e| e.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>())
                .map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<Vec<i64>>>>()?;

    // Stitch chunks inserting the image token; preserve a leading BOS if present
    let mut ids = Vec::new();
    let mut offset = 0;
    if !chunks.is_empty()
        && !chunks[0].is_empty()
        && chunks[0][0] == llava_cfg.bos_token_id as i64
    {
        offset = 1;
        ids.push(chunks[0][0]);
    }
    let image_sep: Vec<i64> = std::iter::repeat(image_token_index).take(offset + 1).collect();

    fn dup_vec<T: Clone>(v: &[T], n: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(v.len() * n);
        for _ in 0..n {
            out.extend_from_slice(v);
        }
        out
    }

    let mut stitched: Vec<Vec<i64>> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 {
            stitched.push(image_sep.clone());
        }
        stitched.push(chunk.clone());
    }
    for seq in stitched {
        // drop duplicated BOS for mid-chunks
        ids.extend(if !seq.is_empty() { seq[1..].to_vec() } else { vec![] });
    }

    let input_len = ids.len();
    Tensor::from_vec(ids, (1, input_len), &Device::Cpu).map_err(anyhow::Error::msg)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device_from_args(args.gpu)?;

    // HF Hub handle to fetch files when --hf is used.
    let api = Api::new()?.model(args.model_path.clone());

    // ------- Load configs, tokenizer, and image preprocessor -------
    let (llava_cfg, tokenizer, clip_vision_cfg, image_processor) = if args.hf {
        let cfg_bytes = std::fs::read(api.get("config.json")?)?;
        let gen_bytes = std::fs::read(api.get("generation_config.json")?)?;
        let pre_bytes = std::fs::read(api.get("preprocessor_config.json")?)?;
        let hf_llava: HFLLaVAConfig = serde_json::from_slice(&cfg_bytes)?;
        let hf_gen: HFGenerationConfig = serde_json::from_slice(&gen_bytes)?;
        let hf_pre: HFPreProcessorConfig = serde_json::from_slice(&pre_bytes)?;
        let llava_cfg = hf_llava.to_llava_config(&hf_gen, &hf_pre);
        let tokenizer_path = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let clip_vision_cfg = hf_llava.to_clip_vision_config();
        (
            llava_cfg,
            tokenizer,
            Some(clip_vision_cfg),
            candle_examples::image_proc::ImageProcessor::from_hf_preprocessor_config(&hf_pre),
        )
   

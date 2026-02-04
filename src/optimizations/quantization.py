"""Model quantization utilities."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.quantization as quant


def quantize_model(
    model: nn.Module,
    quantization_type: str = 'dynamic',
    dtype: torch.dtype = torch.qint8,
    backend: str = 'fbgemm'
) -> nn.Module:
    """Quantize model for inference.
    
    Args:
        model: Model to quantize
        quantization_type: Type of quantization ('dynamic', 'static', or 'qat')
        dtype: Quantization dtype (torch.qint8 or torch.float16)
        backend: Quantization backend ('fbgemm' for CPU, 'qnnpack' for mobile)
        
    Returns:
        Quantized model
    """
    if quantization_type == 'dynamic':
        # Dynamic quantization (quantize weights, activations quantized at runtime)
        model = quant.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=dtype
        )
    elif quantization_type == 'static':
        # Static quantization (requires calibration data)
        model.eval()
        model.qconfig = quant.get_default_qconfig(backend)
        quant.prepare(model, inplace=True)
        # Note: You need to calibrate the model with representative data
        # model = quant.convert(model, inplace=True)
    elif quantization_type == 'qat':
        # Quantization-aware training
        model.train()
        model.qconfig = quant.get_default_qat_qconfig(backend)
        quant.prepare_qat(model, inplace=True)
    
    return model


def prepare_qat(
    model: nn.Module,
    backend: str = 'fbgemm'
) -> nn.Module:
    """Prepare model for quantization-aware training.
    
    Args:
        model: Model to prepare
        backend: Quantization backend
        
    Returns:
        Prepared model
    """
    model.train()
    model.qconfig = quant.get_default_qat_qconfig(backend)
    quant.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """Convert prepared model to quantized model.
    
    Args:
        model: Prepared model
        
    Returns:
        Quantized model
    """
    model.eval()
    return quant.convert(model, inplace=True)


def quantize_to_fp16(model: nn.Module) -> nn.Module:
    """Convert model to FP16 for inference.
    
    Args:
        model: Model to convert
        
    Returns:
        FP16 model
    """
    return model.half()


def quantize_to_int8_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str
):
    """Export model to quantized ONNX format.
    
    Args:
        model: Model to export
        example_input: Example input tensor
        output_path: Path to save ONNX model
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Export to ONNX
        onnx_path = output_path.replace('.onnx', '_fp32.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            opset_version=13,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'logits': {0: 'batch_size', 1: 'seq_len'}
            }
        )
        
        # Quantize ONNX model
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
    except ImportError:
        raise ImportError(
            "onnx and onnxruntime required for ONNX quantization. "
            "Install with: pip install onnx onnxruntime"
        )


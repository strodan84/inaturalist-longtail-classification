import argparse
import torch
import timm


def export_onnx(model_name: str, checkpoint_path: str, num_classes: int, output_path: str):
    device = torch.device("cpu")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Successfully exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="convnext_small")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-classes", type=int, default=10000)
    parser.add_argument("--output", type=str, default="./models/model.onnx")
    args = parser.parse_args()

    export_onnx(args.model_name, args.checkpoint, args.num_classes, args.output)
from utils.model import read_pytorch_model_eval
from models.predict_model import predict_model
import argparse


def predict(opt):
    weights, image_path, image_size, device, num_clasess = (
        opt.weights,
        opt.image_path,
        opt.imgsz,
        opt.device,
        opt.num_classes
    )

    model = read_pytorch_model_eval(weights=weights,
                                    num_classes=num_clasess)

    prediction = predict_model(image=image_path,
                               model=model,
                               img_size=image_size)
    
    print(f"prediction: {prediction}")


def main(parameters):
    predict(parameters)


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--image-path", type=str, default=None, help="image path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--num-classes", default="3", help="num of classes", type=int)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
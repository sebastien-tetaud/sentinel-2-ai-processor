# import argparse
# import yaml

# from model_zoo.models import load_model
# from data.transform import preprocess
# from inference.postprocessing import postprocess
# from utils.visualization import plot_results
# from results.metrics import calculate_metrics

# def parse_args():
#     parser = argparse.ArgumentParser(description="Inference Pipeline for Earth Observation Model")
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         required=True, 
#         help="Path to the configuration YAML file."
#     )
#     return parser.parse_args()

# def load_config(config_path):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#     return config

# def main():
#     args = parse_args()
#     config = load_config(args.config)

#     model = load_model(config["model"]["checkpoint_path"])
#     inputs = preprocess(config["data"]["input_path"])
#     preds = model(inputs)
#     outputs = postprocess(preds)

#     plot_results(outputs, config["visualization"]["save_dir"])
    
#     if config.get("metrics", {}).get("calculate", False):
#         metrics = calculate_metrics(outputs, config["data"]["ground_truth_path"])
#         print("Metrics:", metrics)

# if __name__ == "__main__":
#     main()

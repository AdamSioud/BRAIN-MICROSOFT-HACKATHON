import os
import pandas as pd
import datetime as datetime
from flow_forecast.flood_forecast.trainer import train_function
import wandb

#os.chdir('flow-forecast')
df = pd.read_csv("train.csv")
df["day_of_week"] = df["5 minutes"].map(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%M').weekday())
df["datetime"] = df['5 Minutes']
df.to_csv('train.csv')


def make_config_file(file_path, train_end, valid_end):
  run = wandb.init(project="library_demos")
  wandb_config = wandb.config
  config_default = {
      "model_name": "MultiAttnHeadSimple",
      "model_type": "PyTorch",
      "model_params": {
          "number_time_series": 2,
          "seq_len": wandb_config[
              "forecast_history"
          ],
          "output_seq_len": wandb_config[
              "out_seq_length"
          ],
          "forecast_length": wandb_config[
              "out_seq_length"
          ]
      },
      "dataset_params": {
          "class": "default",
          "training_path": file_path,
          "validation_path": file_path,
          "test_path": file_path,
          "batch_size": wandb_config[
              "batch_size"
          ],
          "forecast_history": wandb_config[
              "forecast_history"
          ],
          "forecast_length": wandb_config[
              "out_seq_length"
          ],
          "train_end": train_end,
          "valid_start": int(train_end+1),
          "valid_end": int(valid_end),
          "test_start": int(valid_end) + 1,
          "target_col": [
              "Lane 1 Flow (Veh/5 Minutes)"
          ],
          "relevant_cols": [
              "Lane 1 Flow (Veh/5 Minutes)",
              "day_of_week"
          ],
          "scaler": "StandardScaler",
          "interpolate": False
      },
      "training_params": {
          "criterion": "MSE",
          "optimizer": "Adam",
          "optim_params": {},
          "lr": wandb_config[
              "lr"
          ],
          "epochs": 10,
          "batch_size": wandb_config[
              "batch_size"
          ]
      },
      "GCS": False,
      "sweep": True,
      "wandb": False,
      "forward_params": {},
      "metrics": [
          "MSE"
      ],
      "inference_params": {
          "datetime_start": "2016-02-24",
          "hours_to_forecast": 150,
          "test_csv_path": file_path,
          "decoder_params": {
              "decoder_function": "simple_decode",
              "unsqueeze_dim": 1
          },
          "dataset_params": {
              "file_path": file_path,
              "forecast_history": wandb_config[
                  "forecast_history"
              ],
              "forecast_length": wandb_config[
                  "out_seq_length"
              ],
              "relevant_cols": [
                  "Lane 1 Flow (Veh/5 Minutes)",
                  "day_of_week"
              ],
              "target_col": [
                  "Lane 1 Flow (Veh/5 Minutes)"
              ],
              "scaling": "StandardScaler",
              "interpolate_param": False
          }
      }
  }
  wandb.config.update(config_default)
  return config_default

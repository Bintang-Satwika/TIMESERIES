import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def scale_array(arr, new_min=-1, new_max=1):

    orig_min = np.min(arr)
    orig_max = np.max(arr)
    scaled_arr = (arr - orig_min) / (orig_max - orig_min) * (new_max - new_min) + new_min
    return scaled_arr, orig_min, orig_max

def inverse_scale_array(scaled_arr, orig_min, orig_max, new_min=-1, new_max=1):

    original_arr = (scaled_arr - new_min) / (new_max - new_min) * (orig_max - orig_min) + orig_min
    return original_arr

def make_windows(data_x,data_y, total_window_size, input_slice, labels_slice):
    x = []
    y = []
    for i in range(len(data_x) - total_window_size + 1):
        window_x = data_x[i:i+total_window_size]
        x.append(window_x[input_slice])

    for i in range(len(data_y) - total_window_size + 1):
        window_y= data_y[i:i+total_window_size]
        y.append(window_y[labels_slice])

    return np.array(x), np.array(y)

def make_windows_autoregressive(data_x, data_y, input_width=48, label_width=1):
    """
    Membuat windowing autoregressive dengan data_x dan data_y yang dipisah.
    
    Parameter:
      data_x       : Array NumPy input dengan shape (total_timesteps, num_features_x)
      data_y       : Array NumPy label dengan shape (total_timesteps, num_features_y)
      input_width  : Jumlah timestep untuk input (default: 48)
      label_width  : Jumlah timestep untuk label (default: 1)
      
    Menghasilkan:
      x_windows    : Array dengan shape (num_windows, input_width, num_features_x)
      y_windows    : Array dengan shape (num_windows, label_width, num_features_y)
    """
    total_window_size = input_width + label_width
    x_windows = []
    y_windows = []
    
    n = len(data_x)  # Asumsi: len(data_x) == len(data_y)
    for i in range(n - total_window_size + 1):
        x_window = data_x[i : i + input_width]
        y_window = data_y[i + input_width : i + total_window_size]
        x_windows.append(x_window)
        y_windows.append(y_window)
    
    return np.array(x_windows), np.array(y_windows)

def process_predictions(model, x_scaled, actual_df, input_width, label_width, orig_min, orig_max):
    predictions = model.predict(x_scaled[slice(None, None, label_width), :, :])
    predictions_reshaped = predictions.reshape(-1,)
    predictions_unscaled = inverse_scale_array(predictions_reshaped, orig_min=orig_min, orig_max=orig_max)
    return pd.DataFrame(predictions_unscaled, index=actual_df.index[input_width:])


def compute_metrics(actual_df, predictions_df):
    mae = mean_absolute_error(actual_df, predictions_df)
    mse = mean_squared_error(actual_df, predictions_df)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_df - predictions_df) / actual_df)) * 100
    return mae, mape, mse, rmse


def compute_error( x_scaled, actual_df, columns, model, input_width, label_width, orig_min, orig_max):
    
    # Process predictions for training and validation sets
    predictions_df = process_predictions(model, x_scaled, actual_df, input_width, label_width, orig_min, orig_max)

    df_predact= pd.concat((predictions_df, actual_df[input_width:]), axis=1).rename(columns={0: 'Prediksi', 'Beban': 'Aktual'})
    
    mae, mape, mse, rmse = compute_metrics(df_predact['Aktual'], df_predact['Prediksi'])
    
    print(f"MAE: {mae}, MAPE %: {mape}, MSE: {mse}, RMSE: {rmse}")
    return df_predact
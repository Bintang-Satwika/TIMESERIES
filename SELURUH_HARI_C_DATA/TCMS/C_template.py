import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def scale_array(arr, new_min=0, new_max=1, orig_min=None, orig_max=None):
    if orig_min is None:
        orig_min = np.min(arr)
        print(f"Computed orig_min: {orig_min}")
    if orig_max is None:
        orig_max = np.max(arr)
        print(f"Computed orig_max: {orig_max}")
    scaled_arr = (arr - orig_min) / (orig_max - orig_min) * (new_max - new_min) + new_min
    return scaled_arr, orig_min, orig_max

def inverse_scale_array(scaled_arr, orig_min, orig_max, new_min=0, new_max=1):

    original_arr = (scaled_arr - new_min) / (new_max - new_min) * (orig_max - orig_min) + orig_min
    return original_arr

def make_multi_input_windows(data_x, data_y, total_window_size, input_slice, labels_slice, periodic_indices):
    """
    data_x: All features (9 features including load, PCA, and periodic codes)
    data_y: Target labels (Load only)
    total_window_size: 384
    periodic_indices: List of column indices for sin_hour, cos_hour, etc.
    """
    x_history = []
    x_periodic = []
    y_target = []

    # The loop ensures we have enough data for the full window
    for i in range(len(data_x) - total_window_size + 1):
        
        # Grab the entire window for both X and Y at once
        window_x = data_x[i : i + total_window_size]
        window_y = data_y[i : i + total_window_size]
        
        # 1. The Historical Array (Orange Input)
        # Slices the first 336 steps of all 9 features
        x_history.append(window_x[input_slice])

        # 2. The Future Periodic Array (Green Input)
        # Slices the next 48 steps of ONLY the periodic coding features
        x_periodic.append(window_x[labels_slice, periodic_indices])

        # 3. The Target Array
        # Slices the next 48 steps of the actual true Load
        y_target.append(window_y[labels_slice])

    return np.array(x_history), np.array(x_periodic), np.array(y_target)

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
    x_train, x_train_periodic = x_scaled
    predictions = model.predict([x_train[slice(None, None, label_width), :, :], x_train_periodic[slice(None, None, label_width), :, :]] )
    predictions_reshaped = predictions.reshape(-1,)
    predictions_unscaled = inverse_scale_array(predictions_reshaped, orig_min=orig_min, orig_max=orig_max)
    limit = len(predictions_unscaled)
    validation_index = actual_df.index[input_width : input_width + limit]
    
    return pd.DataFrame(predictions_unscaled, index=validation_index)


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
    print("before dropna")
    print(df_predact)
    df_predact.dropna(inplace=True)
    print("after dropna")
    print(df_predact)
    mae, mape, mse, rmse = compute_metrics(df_predact['Aktual'], df_predact['Prediksi'])
    
    print(f"MAE: {mae}, MAPE %: {mape}, MSE: {mse}, RMSE: {rmse}")
    return df_predact
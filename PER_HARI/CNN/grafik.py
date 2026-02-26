import json
import matplotlib.pyplot as plt

# Load the data from your json file
with open('PER_HARI\CNN\loss_history_CNN_monday_20.h5_end.json', 'r') as file:
    history = json.load(file)

# Extract loss values
training_loss = history['loss']
validation_loss = history['val_loss']

# Create a range for the epochs starting from 1
epochs = range(1, len(training_loss) + 1)

# Plotting the data
plt.plot(epochs, training_loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')

# Adding labels and title
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
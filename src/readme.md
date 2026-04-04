# Data
Features used for prediction:
- gage_height
- discharge
- gage_diff (difference between consecutive gage measurements)

# Model
TCN:
- 3 input channels (gage_height, discharge, gage_diff)
- Hidden size: 16
- Kernel size: 2
- Output: predicted gage height for the next time step
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate: 0.0003)
- Sequence length: 72 (using the past 72 gage height measurements)

# Training
- Number of epochs: 10   
- Batch size: 32   
- Training data: 80% of the dataset   
- Validation data: 20% of the dataset   

# Evaluation
After training, the model predicts:   
- Gage heights   
- Flash flood probabilities (computed using a sigmoid relative to the threshold)

# Interpreting Outputs
Predicted gage heights: the models water level at the next time step.   
Flash flood probability (%): how likely it is that the gage height exceeds the 2.2 ft threshold.
- 0% → very unlikely to flood
- 50% → near threshold
- 100% → definitely above threshold (flash flood)

# Usage
python flood_prediction.py <gage_height_data.csv> <discharge_data.csv>

# Notes
- Just uses the normal csv files and merges gage height and discharge data based on timestamps. 
- Uses a threshold of 2.2 feet for a flash flood. This is taken from the peaks data and is just the lowest point for now. 
- Most of the predictions are 0.002% because of how little the data fluctuates. It may be beneficial for us to introduce artificial flash flood points to allow the model to learn and predict better. 
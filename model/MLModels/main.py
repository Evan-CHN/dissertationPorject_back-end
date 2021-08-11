from backend.model.MLModels.LSTM.LSTMToolKit import LSTMToolKit
from backend.model.MLModels.RNN.RNNToolKit import RNNToolKit

if __name__ == '__main__':
    cnn = LSTMToolKit()
    cnn.text_processed()
    cnn.train_model()
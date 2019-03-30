import pickle


def preprocess_data(data_path, save_path):
    data = []
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(data_path, labels_path = None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if labels_path:
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        return data, labels

    return data

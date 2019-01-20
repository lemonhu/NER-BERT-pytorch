"""split the msra dataset for our model and build tags"""
import os
import random


def load_dataset(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset) as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                word, tag = line.split('\t')
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    """Write sentences.txt and tags.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print('Saving in {}...'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
        open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print('- done.')

def build_tags(data_dir, tags_file):
    """Build tags from dataset
    """
    data_types = ['train', 'val', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    with open(tags_file, 'w') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    # Check that the dataset exist, two balnk lines at the end of the file
    path_train_val = 'data/msra/msra_train_bio'
    path_test = 'data/msra/msra_test_bio'
    msg = '{} or {} file not found. Make sure you have downloaded the right dataset'.format(path_train_val, path_test)
    assert os.path.isfile(path_train_val) and os.path.isfile(path_test), msg

    # Load the dataset into memory
    print('Loading MSRA dataset into memory...')
    dataset_train_val = load_dataset(path_train_val)
    dataset_test = load_dataset(path_test)
    print('- done.')

    # Make a list that decides the order in which we go over the data
    order = list(range(len(dataset_train_val)))
    random.seed(2019)
    random.shuffle(order)

    # Split the dataset into train, val(split with shuffle) and test
    train_dataset = [dataset_train_val[idx] for idx in order[:42000]]  # 42000 for train
    val_dataset = [dataset_train_val[idx] for idx in order[42000:]]  # 3000 for val
    test_dataset = dataset_test  # 3442 for test
    save_dataset(train_dataset, 'data/msra/train')
    save_dataset(val_dataset, 'data/msra/val')
    save_dataset(test_dataset, 'data/msra/test')

    # Build tags from dataset
    build_tags('data/msra', 'data/msra/tags.txt')


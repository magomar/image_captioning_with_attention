class Config(object):
    """ Wrapper class for various (hyper)parameters.
    
    """

    def __init__(self):
        # about the model architecture
        self.cnn = 'inception-v3'               # 'inception-v3', vgg16' or 'resnet50'
        self.rnn = 'gru'                        #  'gru' or 'lstm'
        self.max_caption_length = 20
        self.embedding_dim = 256
        self.rnn_units = 512

        # about the weight initialization and regularization
        self.weight_initilization_method = 'glorot'     # 'glorot', 'xavier', etc.

        # about the optimization
        self.num_epochs = 20
        self.batch_size = 64
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.loss = 'sparse_categorical_crossentropy'

        # about the dataset
        self.dataset_name = 'COCO_2014'
        self.num_examples =  10000
        self.buffer_size = 1000
        self.drop_remainder = True

        # about the saver (checkpoint manager)
        # self.save_period = 1000
        self.max_checkpoints = 40
        self.checkpoints_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        # self.vocabulary_file = './data/vocabulary.csv'
        self.max_vocabulary_size = 5000

        # about image features
        self.extract_image_features = False
        self.image_features_batchsize = 16
        self.image_features_dir = './features/'

        # about the training
        self.train_image_dir = './data/coco/train2014/'
        self.train_caption_file = './data/coco/annotations/captions_train2014.json'
        # self.temp_annotation_file = './temp/train/anns.csv'
        # self.temp_data_file = './temp/train/data.npy'

        # about the evaluation
        self.eval_image_dir = './data/coco/val2014/'
        self.eval_caption_file = './data/coco/annotations/captions_val2014.json'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'
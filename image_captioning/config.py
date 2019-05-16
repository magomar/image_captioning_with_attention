class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'inception-v3'               # 'inception-v3', vgg16' or 'resnet50'
        self.rnn = 'gru'                        #  'gru' or 'lstm'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_rnn_units = 512

        # about the weight initialization and regularization
        self.weight_initilization_method = 'glorot'     # 'glorot', 'xavier', etc.

        # about the optimization
        self.num_epochs = 20
        self.batch_size = 64
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'

        # about the dataset
        self.buffer_size = 1000
        self.drop_remainder = False

        # about the saver (checkpoint manager)
        self.save_period = 1000
        self.max_chekpoints = 5
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        # self.vocabulary_file = './data/vocabulary.csv'
        self.vocabulary_size = 5000

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
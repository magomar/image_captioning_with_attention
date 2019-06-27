class Config(object):
    """ Wrapper class for various (hyper)parameters.
    
    """

    def __init__(self):
        # general
        self.log_dir = './log/'  # 'log' or None

        # about the model architecture
        self.cnn = 'inception_resnet_v2'  #'vgg16', 'inception_v3', 'xception, 'resnet50', 'nasnet_large','inception_resnet_v2'
        self.rnn = 'lstm'       #  'gru' or 'lstm'
        self.embedding_dim = 256
        self.rnn_units = 512
        self.num_features = 256
        self.use_attention = True
        
        # about the weight initialization and regularization
        self.weight_initialization = 'glorot_uniform' # 'glorot', 'xavier', etc.
        self.dropout = 0.5

        # about the optimization
        self.num_epochs = 16
        self.batch_size = 64
        self.optimizer = 'Nadam'  # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        # self.loss = 'sparse_categorical_crossentropy'

        # about the dataset
        self.dataset_name = 'COCO_2014'
        self.buffer_size = 1000
        self.drop_remainder = True
        self.limit_length = True
        self.max_length = 25

        # about the saver (checkpoint manager)
        self.max_checkpoints = 10 # max number of checkpoints to keep 
        self.checkpoints_frequency = 1 # number of epochs before saving checkpoint
        self.checkpoints_dir = './models/checkpoints/'
        self.summary_dir = './summary/' 

        # about the vocabulary
        self.vocabulary_file = './data/vocabulary.pickle'
        self.vocabulary_size = 10000

        # about image features
        self.extract_image_features = True
        self.image_features_batchsize = 16
        self.image_features_dir = './feature_maps_cache/'

        # about the training
        self.resume_from_checkpoint = True
        self.num_train_examples =  None
        self.train_image_dir = './data/coco/train2014/'
        self.train_captions_file = './data/coco/annotations/captions_train2014.json'

        # about the evaluation
        self.beam_width = 3
        self.use_beam_search = False
        self.normalize_by_length = True
        self.eval_image_dir = './data/coco/val2014/'
        self.eval_captions_file = './data/coco/annotations/captions_val2014.json'
        self.eval_result_dir = './results/eval/images/'
        self.eval_result_file = './results/eval/eval_results.json'
        self.save_eval_result_as_image = False

        # about the inference
        self.test_image_dir = './results/test/images/'
        self.test_result_dir = './results/test/'
        self.test_result_file = './results/test_results.csv'
import cntk as C
import numpy as np
import argparse
# TODO: add argpars for model

data_dict = {
    'toy': ( 943, 26),
    'ag': (90258, 5)
} 

data_name = {
    'toy':('atis.train.ctf', 'atis.test.ctf'),
    'ag' :('ag.train.ctf', 'ag.train.ctf')
}

feature_dict = {
    'toy':('S0', 'S1'),
    'ag' :('word', 'class')
}

epoch_size_dict = {
    'toy':  18000, # 18000 is half of the data
    'ag': 119998
}
lr = 3e-2
max_epoch = 10
batch_size = 300
epoch_size = 119998

num_vocabs = 90258
num_labels = 5

x = C.sequence.input_variable(num_vocabs)
y = C.input_variable(num_labels)

def create_reader(path, is_training, feature_name, label_name):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field=feature_name, shape=num_vocabs,  is_sparse=True),
         intent        = C.io.StreamDef(field=label_name, shape=num_labels, is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


def create_birnn(runit_forward,runit_backward, name=''):
    with C.layers.default_options(initial_state=0.1 ):
        negRnn = C.layers.Fold(runit_backward, go_backwards=True)
        posRnn = C.layers.Fold(runit_forward, go_backwards=False)
    @C.Function
    def BiRnn(e):
        h = C.splice(posRnn(e), negRnn(e), name=name)
        return h
    return BiRnn


def build_graph( self_attention, self_penalty, embeded_dim = 60, h_dim = 150, d_a = 350, r = 30):
    
    with C.layers.default_options(init = C.xavier()):
        embeded = C.layers.Embedding(embeded_dim)(x )
        embeded = C.layers.Stabilizer()(embeded)

        H = create_birnn(C.layers.GRU(h_dim), C.layers.GRU(h_dim))(embeded)

        if self_attention :
            Ws1 = C.parameter(shape=(d_a, 2 * h_dim), name="Ws1")
            Ws2 = C.parameter(shape=(r, d_a), name="Ws2")
            A = C.softmax( C.times( Ws2, C.tanh(C.times_transpose(Ws1, H)) ))
            H = C.times(A, H)# the M in the paper

            if self_penalty :
                I = C.constant(np.eye(r), dtype = np.float32)
                P = C.times_transpose(A, A) - I# r*r
                p = C.reduce_sum(C.abs(C.element_times(P, P) )) # frobenius norm **2

        y_ = C.layers.Dense(200, activation = C.ops.relu )(H)
        # y_pre = C.layers.Dense(num_labels, activation = None)(y_)
        def selfAtt(x):
            y_pre = C.layers.Dense(num_labels, activation = None)(y_)
            return y_pre
        if self_penalty:
            selfAtt.p = p
        return selfAtt
    

def create_criterion_function(model, y_pre, labels, self_penalty):
    loss = C.cross_entropy_with_softmax(y_pre, labels)
    if self_penalty:
        p_coefficient = 1
        loss += model.p * p_coefficient
    errs = C.classification_error(y_pre, labels)
    return loss, errs # (model, labels) -> (loss, error metric)

def train( model, reader):
    y_pre = model(x)
    loss, label_error = create_criterion_function(model, y_pre, y, True)
    lr_per_minibatch = [lr] + [lr/2] + [lr/4]
    # lr_per_minibatch = [lr * batch_size for lr in lr_per_sample]
    
    lr_schedule = C.learning_parameter_schedule(lr_per_minibatch, epoch_size=epoch_size)

    # Momentum schedule
    momentums = C.momentum_schedule(0.9048374180359595, minibatch_size=batch_size)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epoch)
    # learner = C.sgd(model.parameters, lr_schedule)
    learner = C.adam( y_pre.parameters, lr_schedule, momentum = momentums, gradient_clipping_threshold_per_sample=15)
    trainer = C.Trainer(y_pre, (loss, label_error), learner, progress_printer)# []

    C.logging.log_number_of_parameters(y_pre)# print # parameters and # tensor

    loss_summary = []
    step = 0
    data_map={x: reader.streams.query, y: reader.streams.intent}

    t = 0
    for epoch in range(max_epoch):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(batch_size, input_map= data_map)  # fetch minibatch
            # print(data)
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples      
            if t % 6000 == 0:
                training_loss = trainer.previous_minibatch_loss_average
                error = trainer.previous_minibatch_evaluation_average
                print("epoch: {}, step: {}, loss: {:.5f}, error {:.5f}".format(epoch, t, training_loss, error))
        trainer.summarize_training_progress()
        # do_test()

def evaluate(model, reader):
    # Create the loss and error functions
    loss, label_error = create_criterion_function(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)
    
    # Assign the data fields to be read from the input
    
    data_map={x: reader.streams.query, y: reader.streams.intent} 

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
        if not data:                                 # until we hit the end
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
     
    evaluator.summarize_test_progress()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-2, help='Learning rate')
    parser.add_argument('--dataset', default='ag', help='The dataset you choose, should be "ag" or "toy" ')
    parser.add_argument('--max_epoch', default=5, type=int, help='Max epoches')
    parser.add_argument('--batch_size', default=300, type=int, help='Minibatch size')
    parser.add_argument('--self_attention', action='store_true', help='Whether to use selfAttention')
    args = parser.parse_args()
    
    global num_vocabs, num_labels
    num_vocabs, num_labels = data_dict[args.dataset]
    train_data, test_data = data_name[args.dataset]
    feature_name, label_name = feature_dict[args.dataset]

    global max_epoch, batch_size, epoch_size
    max_epoch = args.max_epoch
 
    batch_size = args.batch_size

    epoch_size = epoch_size_dict[args.dataset]

    train_reader = create_reader(train_data, True, feature_name, label_name)
    test_reader = create_reader(test_data, False, feature_name, label_name)

    global x, y
    x = C.sequence.input_variable(num_vocabs)
    y = C.input_variable(num_labels)
    model = build_graph( args.self_attention, self_penalty=True)
    
    train( model, train_reader)
    
    evaluate( model, test_reader)

if __name__ == '__main__':
    main()
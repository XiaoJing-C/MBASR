import argparse
import datetime
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation import evaluate_origin

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    parser.add_argument('--dataset', nargs='?', default='UB',
                        help='dataset')
    parser.add_argument('--data', nargs='?', default='datasets/UB/data',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    # parser.add_argument('--l2', default=1e-6, type=float)

    return parser.parse_args()


class GRUnetwork:
    def __init__(self, emb_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.item_num=int(item_num)

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)

        gru_out, self.states_hidden= tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.hidden_size),
            self.input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )
        # Add dropout
        with tf.name_scope("dropout"):
            self.state_hidden = tf.layers.dropout(self.states_hidden,
                                     rate=args.dropout_rate,
                                      seed=args.random_seed,
                                      training=tf.convert_to_tensor(self.is_training))

        self.output = tf.contrib.layers.fully_connected(self.states_hidden,self.item_num,activation_fn=tf.nn.softmax,scope='fc')
        # self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(args.l2),
        #                                                   tf.trainable_variables())
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target, self.output)
        # self.loss = tf.reduce_mean(self.loss + self.reg)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.emb_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.emb_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        all_embeddings['item_embeddings']=item_embeddings
        return all_embeddings

if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]

    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    GRUnet = GRUnetwork(emb_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

    saver = tf.train.Saver(max_to_keep=10000)

    nowTime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    save_dir = 'JD/model/gru4rec/emb_{}_drop_ratio_{}_seed_{}_{}'.format(args.emb_size,args.dropout_rate,args.random_seed,nowTime)
    save_dir = os.path.join(args.dataset,save_dir)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)
    print(save_dir)
    data_loader = pd.read_pickle(os.path.join(data_directory, 'train.df'))
    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step=0
    # with tf.Session() as sess:
        # # Initialize variables
        # sess.run(tf.global_variables_initializer())
        # num_rows=data_loader.shape[0]
        # num_batches=int(num_rows/args.batch_size)
        # print(num_rows,num_batches)
        # best_hit_5 = -1
        # count = 0
        # for i in range(args.epoch):
        #     print(i)
        #     start_time_i = datetime.datetime.now()

        #     for j in range(num_batches):
        #         batch = data_loader.sample(n=args.batch_size).to_dict()
        #         item_seq = list(batch['item_seq'].values())
        #         len_seq = list(batch['len_seq'].values())
                
        #         # item_seq = [row[-5::-5][::-1]+[item_num]*40 for row in item_seq]
        #         # len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                
        #         # len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                
        #         target = list(batch['target'].values())
        #         loss, _ = sess.run([GRUnet.loss, GRUnet.opt],
        #                            feed_dict={GRUnet.item_seq: item_seq,
        #                                       GRUnet.len_seq: len_seq,
        #                                       GRUnet.target: target,
        #                                       GRUnet.is_training:True
        #         })
        #         total_step+=1
        #         # if total_step % 200 == 0:
        #         #     print("the loss in %dth batch is: %f" % (total_step, loss))
        #         # if total_step % 2000 == 0:
        #     over_time_i = datetime.datetime.now()
        #     total_time_i = (over_time_i - start_time_i).total_seconds()
        #     print('total times: %s' % total_time_i)

        #     hit5, ndcg5,hit10,ndcg10,hit20,ndcg20 = evaluate_origin(sess,GRUnet,data_directory,topk,have_dropout=True,have_user_emb=False,is_test=False)
        #     if hit5 > best_hit_5 :
        #         best_hit_5 = hit5
        #         count = 0
        #         save_root = os.path.join(save_dir,
        #                                  'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
        #                                      i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
        #         isExists = os.path.exists(save_root)
        #         if not isExists:
        #             os.makedirs(save_root)
        #         model_name = 'gru4rec.ckpt'
        #         save_root = os.path.join(save_root, model_name)
        #         saver.save(sess, save_root)
        #     else:
        #         count += 1
        #     if count == args.early_stop_epoch:
        #         break

    with tf.Session() as sess :
        saver.restore(sess, '/home/temp_user/xiaoj/SHOCCF-baselines/UB/model/gru4rec/emb_64_drop_ratio_0.5_seed_0_20231228T221308/epoch_5_hit@5_0.0278_ndcg@5_0.0172_hit@10_0.0436_ndcg@10_0.0223_hit@20_0.0692_ndcg@20_0.0287/gru4rec.ckpt')
        hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, GRUnet, data_directory, topk,
                                                                    have_dropout=True,have_user_emb=False,
                                                                    is_test=True, type='all')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, GRUnet, data_directory, topk,
    #                                                                 have_dropout=True,
    #                                                                 is_test=True,type='clicked')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, GRUnet, data_directory, topk,
    #                                                                 have_dropout=True,
    #                                                                 is_test=True,type='unclicked')
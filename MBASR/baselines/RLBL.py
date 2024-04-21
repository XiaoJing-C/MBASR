import argparse
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from evaluation import evaluate_RLBL
from augmentation import augmentation
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='datasets/Tmall/data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--window_size', type=int, default=5, help='Length of windows.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=0.8, help='sub.')
    parser.add_argument('--gamma', type=float, default=0.4, help='del')
    parser.add_argument('--beta', type=float, default=0.4, help='reorder.')
    parser.add_argument('--lamda', type=float, default=0.4, help='swap behaivor.')
    parser.add_argument('--tag', type=int, default=2, help='1->del 2->sub 3->reorder')
    return parser.parse_args()
 
class RLBL:
    def __init__(self, emb_size,learning_rate,item_num,user_num,seq_len,window_size=5):
        '''
        :param emb_size: the dimensionality of latent vector
        :param learning_rate: learning_rate  
        :param item_num: item_num
        :param seq_len: the length of the input sequence
        :param window_size: the window size of RLBL
        '''
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.behavior_num = 2
        self.item_num=int(item_num)
        self.user_num=int(user_num)
        self.window_size = window_size
        # 
        self.all_embeddings=self.initialize_embeddings()
        # 
        self.uid_seq = tf.placeholder(tf.int32, [None],name='uid_seq')
        self.item_seq = tf.placeholder(tf.int32, [None, seq_len],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.position = tf.placeholder(tf.int32, [None, seq_len],name='position')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.behavior_seq = tf.placeholder(tf.int32, [None, seq_len])
        self.target_behavior = tf.placeholder(tf.int32, [None])
        self.targetM = tf.nn.embedding_lookup(self.all_embeddings['behavior_specific_matrics'], self.target_behavior)

        # 
        self.M = tf.nn.embedding_lookup(self.all_embeddings['behavior_specific_matrics'], self.behavior_seq)
        self.C = tf.nn.embedding_lookup(self.all_embeddings['position_specific_matrics'], self.position)
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)
        self.x = tf.matmul(tf.matmul(self.C,self.M),tf.reshape(self.input_emb,[-1,self.seq_len,self.emb_size,1]))

        # 
        self.x = tf.reshape(self.x,[-1,self.seq_len,self.emb_size])
        real_len = int(self.seq_len // self.window_size)
        self.x = tf.split(self.x,num_or_size_splits=real_len,axis=1)
        for i in range(real_len):
            self.x[i] = tf.reduce_sum(self.x[i],axis=1,keep_dims=True)
        self.x = tf.concat(self.x,axis=1)
        self.new_input_emb = self.x

        # 
        _, self.h= tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.BasicRNNCell(self.emb_size),
            self.new_input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )

        # 
        self.user_emb = tf.nn.embedding_lookup(self.all_embeddings['user_embeddings'],self.uid_seq)
        self.final_h = self.h + self.user_emb
        self.final_h = tf.reshape(
                        tf.matmul(tf.reshape(self.final_h, [-1, 1, self.emb_size]), self.targetM),
                        [-1, self.emb_size])

        with tf.name_scope("dropout"):
            self.final_h = tf.layers.dropout(self.final_h,
                                   rate=args.dropout_rate,
                                   seed=args.random_seed,
                                   training=tf.convert_to_tensor(self.is_training))
        self.output = tf.contrib.layers.fully_connected(self.final_h,self.item_num,activation_fn=tf.nn.softmax,scope='fc')
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target, self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        user_embeddings= tf.Variable(tf.random_normal([self.user_num, self.hidden_size], 0.0, 0.01),
            name='user_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        user_embeddings = tf.concat([user_embeddings,padding],axis=0)
        behavior_specific_matrics = tf.Variable(tf.random_normal([self.behavior_num,self.emb_size,self.emb_size],0.0,0.01),name="behavior_specific_matrics")
        padding = tf.zeros([1,self.emb_size,self.emb_size],dtype= tf.float32)
        behavior_specific_matrics = tf.concat([behavior_specific_matrics,padding],axis=0)
        position_specific_matrics = tf.Variable(tf.random_normal([self.window_size,self.emb_size,self.emb_size],0.0,0.01),name="position_specific_matrics")
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['user_embeddings']=user_embeddings
        all_embeddings['behavior_specific_matrics']=behavior_specific_matrics
        all_embeddings['position_specific_matrics']=position_specific_matrics
        return all_embeddings

if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma, lamda = args.tag, args.alpha, args.beta, args.gamma, args.lamda
    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_len and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    user_num = data_statis['user_num'][0]
    topk=[5,10,20]
    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    RLBLnet = RLBL(emb_size=args.emb_size, learning_rate=args.lr,item_num=item_num,user_num=user_num,seq_len=state_size,window_size=args.window_size)

    saver = tf.train.Saver(max_to_keep=10000)


    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if tag == 0:
        label = 'origin'
    elif tag == 1 or tag == 11:
        label = gamma
    elif tag == 2:
        label = alpha
    elif tag == 3 or tag == 33:
        label = beta
    elif tag == 4 or tag == 5:
        label = lamda
        
    save_dir = './model/Tmall/newRLBL/7/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)

    # save_dir = './model/RLBL/emb_{}_dropout_{}_{}'.format(args.emb_size,args.dropout_rate,nowTime)

    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train.df'))
    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step=0
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows=data_loader.shape[0]
        num_batches=int(num_rows/args.batch_size)
        print(num_rows,num_batches)
        best_hit_5 = -1
        count = 0
        for i in range(args.epoch):
            print(i)
            start_time_i = datetime.datetime.now()  # 

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                uid_seq = list(batch['uid'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                len_seq = list(batch['len_seq'].values())
              
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                
                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size, tag, alpha, beta, gamma, lamda)

                
                target=list(batch['target'].values())
                target_behavior = list(batch['is_buy'].values())
                
                position_info = np.zeros((args.batch_size,state_size))
                for idx, len_i in enumerate(len_seq):
                    len_seq[idx] = math.ceil(len_i/args.window_size)
                    for s in range(0,len(item_seq[0]),args.window_size):
                        position_info[idx][s:s+args.window_size]=range(0,args.window_size)
                loss, _ = sess.run([RLBLnet.loss, RLBLnet.opt],
                                   feed_dict={RLBLnet.item_seq: item_seq,
                                              RLBLnet.uid_seq: uid_seq,
                                              RLBLnet.len_seq: len_seq,
                                              RLBLnet.behavior_seq : behavior_seq,
                                              RLBLnet.position : position_info,
                                              RLBLnet.target_behavior : target_behavior,
                                              RLBLnet.target: target,
                                              RLBLnet.is_training:True
                })
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))

            over_time_i = datetime.datetime.now()  # 
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            # for k, v in zip(variable_names, values):
            #     print("Variable: ", k)
            #     print("Shape: ", v.shape)
            #     print(v)
            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_RLBL(sess, RLBLnet, data_directory, topk, have_dropout=True, is_test=True, have_user_emb=True)
            if hit5 > best_hit_5:
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'RLBL.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)

            else:
                count += 1
            if count == args.early_stop_epoch:
                break


    # with tf.Session() as sess :
    #     saver.restore(sess, '/home/temp_user/xiaoj/SHOCCF-baselines/model/Tmall/newRLBL/7/tag_1_param_0.2_20231118_163850/epoch_14_hit@5_0.0363_ndcg@5_0.0218_hit@10_0.0596_ndcg@10_0.0293_hit@20_0.0938_ndcg@20_0.0379/RLBL.ckpt')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_RLBL(sess, RLBLnet, data_directory, topk, have_dropout=True, is_test=True, have_user_emb=True, type='all')

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from augmentation import augmentation
from evaluation import evaluate_BAR
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    # parser.add_argument('--dataset', nargs='?', default='ML1M',
    #                     help='dataset')
    parser.add_argument('--data', nargs='?', default='datasets/JD/data',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    # parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--alpha', type=float, default=0.8, help='sub.')
    parser.add_argument('--gamma', type=float, default=0.4, help='del')
    parser.add_argument('--beta', type=float, default=0.4, help='reorder.')
    parser.add_argument('--lamda', type=float, default=0.4, help='swap behaivor.')
    parser.add_argument('--tag', type=int, default=5, help='1->del 2->sub 3->reorder')
    return parser.parse_args()


class GRUnetwork:
    def __init__(self, emb_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.behavior_num = 2
        self.item_num=int(item_num)

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.next_behaviors_input = tf.placeholder(tf.int32, [None])
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])
        self.position = tf.placeholder(tf.int32, [None, state_size])

        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)
        self.next_behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.next_behaviors_input)
        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.position_emb = tf.nn.embedding_lookup(self.all_embeddings['position_embeddings'], self.position)
        self.att_behavior_input = tf.tile(tf.expand_dims(self.next_behavior_emb, axis=1), (1, self.state_size, 1))
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, item_num)), -1)
        self.mix_input = self.position_emb + self.behavior_emb + self.input_emb
        self.att_input = tf.concat([self.mix_input,self.mix_input-self.att_behavior_input,self.mix_input*self.att_behavior_input, self.att_behavior_input], axis=2)

        self.att_net = tf.contrib.layers.fully_connected(self.att_input, self.hidden_size,
                                                         activation_fn=tf.nn.relu, scope="att_net1")
        self.att_net = tf.contrib.layers.fully_connected(self.att_net, 1,
                                                         activation_fn=tf.nn.tanh, scope="att_net2")
        self.att = self.att_net 
        self.new_input_emb = self.input_emb * (1 + self.att)
        
        gru_out, self.states_hidden= tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.emb_size),
            self.new_input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )

        self.state_behavior_feat = tf.contrib.layers.fully_connected(
            tf.concat([self.states_hidden, self.next_behavior_emb], axis=1), self.hidden_size,
            activation_fn=tf.nn.relu, scope="state_behavior_feat")  # all q-values
        self.final_feat = tf.concat([self.states_hidden, self.state_behavior_feat, self.next_behavior_emb], axis=1)

        with tf.name_scope("dropout"):
            self.final_feat = tf.layers.dropout(self.final_feat,
                                     rate=args.dropout_rate,
                                   seed=args.random_seed,
                                   training=tf.convert_to_tensor(self.is_training))

        self.output = tf.contrib.layers.fully_connected(self.final_feat,self.item_num,activation_fn=tf.nn.softmax,scope='fc')

        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        behavior_embeddings = tf.Variable(tf.random_normal([self.behavior_num, self.hidden_size], 0.0, 0.01),
                                          name='behavior_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings,padding],axis=0)
        position_embeddings = tf.Variable(tf.random_normal([self.state_size + 1, self.hidden_size], 0.0, 0.01),
                                          name='position_embeddings')
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        all_embeddings['position_embeddings'] = position_embeddings
        return all_embeddings

if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma, lamda = args.tag, args.alpha, args.beta, args.gamma, args.lamda
    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    tf.reset_default_graph()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)


    GRUnet = GRUnetwork(emb_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)
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
        
    save_dir = './model/JD/newGRU_BAR/2/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)
    
    # save_dir = './model/gru4rec_BAR/emb_size_{}_dropout_{}_{}'.format(args.emb_size,args.dropout_rate,nowTime)
    
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
            start_time_i = datetime.datetime.now()

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                next_behavior = list(batch['is_buy'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())
               
                # len_seq = [len(row) for row in item_seq]
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                
                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                
                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size, tag, alpha, beta, gamma, lamda)
            
                position_info = np.zeros((args.batch_size,state_size))
                for idx, l in enumerate(len_seq):
                    position_info[idx][:l]=range(l,0,-1)
                
                loss, _ = sess.run([GRUnet.loss, GRUnet.opt],
                                   feed_dict={GRUnet.item_seq: item_seq,
                                              GRUnet.len_seq: len_seq,
                                              GRUnet.behavior_seq : behavior_seq,
                                              GRUnet.next_behaviors_input : next_behavior,
                                              GRUnet.position: position_info,
                                              GRUnet.target: target,
                                              GRUnet.is_training:True
                })
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))

            over_time_i = datetime.datetime.now()
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5,hit10,ndcg10,hit20,ndcg20 = evaluate_BAR(sess,GRUnet,data_directory,topk,have_dropout=True,is_test=False)
            if hit5 > best_hit_5 :
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'gru4rec.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)

            else:
                count += 1
            if count == args.early_stop_epoch:
                break


    # with tf.Session() as sess :
    #     saver.restore(sess, 'gru4rec.ckpt')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, GRUnet, data_directory, topk, have_dropout=True, is_test=True)
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, GRUnet, data_directory, topk, have_dropout=True,
    #                                                              have_user_emb=False, is_test=True,type='clicked')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, GRUnet, data_directory, topk, have_dropout=True,
    #                                                              have_user_emb=False, is_test=True,type='unclicked')

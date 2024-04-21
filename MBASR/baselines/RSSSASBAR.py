import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from utility import *
from SASRecModules import *
from augmentation import augmentation
import random
import datetime
from evaluation import evaluate_BAR
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    # parser.add_argument('--dataset', nargs='?', default='datasets/Tmall/data',
    #                     help='dataset')
    parser.add_argument('--data', nargs='?', default='datasets/Tmall/data',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--alpha', type=float, default=0.8, help='sub.')
    parser.add_argument('--gamma', type=float, default=0.4, help='del')
    parser.add_argument('--beta', type=float, default=0.4, help='reorder.')
    parser.add_argument('--lamda', type=float, default=0.4, help='swap behaivor.')
    parser.add_argument('--tag', type=int, default=2, help='1->del 2->sub 3->reorder')

    return parser.parse_args()


class SASRecnetwork:
    def __init__(self, hidden_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = hidden_size
        self.hidden_size = hidden_size
        self.item_num=int(item_num)
        self.behavior_num = 4
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.next_behaviors_input = tf.placeholder(tf.int32, [None])
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])
        self.position = tf.placeholder(tf.int32, [None, state_size])
        self.next_behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.next_behaviors_input)
        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.position_emb = tf.nn.embedding_lookup(self.all_embeddings['position_embeddings'], self.position)

        self.att_behavior_input = tf.tile(tf.expand_dims(self.next_behavior_emb, axis=1), (1, self.state_size, 1))

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, item_num)), -1)
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)

        self.mix_input = self.position_emb + self.behavior_emb + self.input_emb
        self.att_input = tf.concat([self.mix_input,self.mix_input-self.att_behavior_input,self.mix_input*self.att_behavior_input, self.att_behavior_input], axis=2)
        self.att_input*=mask

        self.att_net = tf.contrib.layers.fully_connected(self.att_input, self.hidden_size,
                                                         activation_fn=tf.nn.relu, scope="att_net1")
        self.att_net = tf.contrib.layers.fully_connected(self.att_net, 1,
                                                         activation_fn=tf.nn.tanh, scope="att_net2")
        self.att = self.att_net
        # Positional Encoding
        pos_emb=tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0), [tf.shape(self.item_seq)[0], 1]))
        self.seq=self.input_emb * (1 + self.att) + pos_emb

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, item_num)), -1)
        #Dropout
        self.seq = tf.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     seed=args.random_seed,
                                     training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(self.seq),
                                               keys=self.seq,
                                               num_units=self.hidden_size,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                       dropout_rate=args.dropout_rate,
                                       is_training=self.is_training)
                self.seq *= mask

        self.seq = normalize(self.seq)
        self.state_hidden=extract_axis_1(self.seq, self.len_seq - 1)

        self.state_behavior_feat = tf.contrib.layers.fully_connected(
            tf.concat([self.state_hidden, self.next_behavior_emb], axis=1), self.hidden_size,
            activation_fn=tf.nn.relu, scope="state_behavior_feat")  # all q-values

        self.final_feat = tf.concat([self.state_hidden, self.state_behavior_feat,self.next_behavior_emb], axis=1)

        self.output = tf.contrib.layers.fully_connected(self.final_feat,self.item_num,activation_fn=tf.nn.softmax,scope='fc')

        self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(args.l2), tf.trainable_variables())
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss + self.reg)

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
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
        all_embeddings['pos_embeddings']=pos_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        all_embeddings['position_embeddings'] = position_embeddings
        return all_embeddings

def get_target(input, input_beh, target, next_behavior, item_num, a=0.8):
    sample_target = []
    sample_target_beh = []
    sample_seq = []
    sample_beha = []
    for i in range (len(input)):
        # if i == 43:
        #     print("xxxxxx")
        item_seq = input[i]
        behavior_seq = input_beh[i]
        t = target[i]
        t_b = next_behavior[i]
        length = len(item_seq) + 1
        # sample_len = math.floor(length * 0.6)
        if item_seq[0] == item_num:
            sample_target.append(t)
            sample_target_beh.append(t_b)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            beh_seq = np.pad(behavior_seq, (0, 50 - len(behavior_seq)), 'constant', constant_values=2)
            sample_seq.append(list(item_seq))
            sample_beha.append(list(beh_seq))
            continue
        item_indices = np.arange(length)  #
        item_importance = np.power(a, length - item_indices)
        
        # item_importance = np.exp(item_importance)
        prob = item_importance / np.sum(item_importance)
        target_index = np.random.choice(range(length), size=1, replace=False, p=prob)[0]
        if target_index >= length - 1:
            sample_target.append(t)
            sample_target_beh.append(t_b)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            beh_seq = np.pad(behavior_seq, (0, 50 - len(behavior_seq)), 'constant', constant_values=2)
            sample_seq.append(list(item_seq))
            sample_beha.append(list(beh_seq))
        else:
            sample_target.append(item_seq[target_index])
            sample_target_beh.append(behavior_seq[target_index])
            del item_seq[target_index]
            del behavior_seq[target_index]
            item_seq.append(t)
            behavior_seq.append(t_b)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            beh_seq = np.pad(behavior_seq, (0, 50 - len(behavior_seq)), 'constant', constant_values=2)
            sample_beha.append(list(beh_seq))
            sample_seq.append(list(item_seq))
            
    return sample_target, sample_target_beh, sample_seq, sample_beha

if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma, lamda = args.tag, args.alpha, args.beta, args.gamma, args.lamda
    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)


    SASRec = SASRecnetwork(hidden_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

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
        
    save_dir = './model/Tmall/RSSSASBAR/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)

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
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())
                next_behavior = list(batch['is_buy'].values())
           
                sp_item_seq = item_seq
                sp_beh_seq = behavior_seq
                sp_len_seq = [np.sum(seq!=item_num) for seq in sp_item_seq]
                
                sp_len_seq = [ss if ss > 0 else 1 for ss in sp_len_seq]

                input = [list(sp_item_seq[r][:l1]) for r,l1 in enumerate(sp_len_seq)]
                input_beh = [list(sp_beh_seq[r][:l1]) for r,l1 in enumerate(sp_len_seq)]
                
                sample_target, sample_target_beh, sample_seq, sample_beha = get_target(input, input_beh, target, next_behavior, item_num)
               
                
                position_info = np.zeros((args.batch_size,state_size))
                
                # l_seq = len_seq.copy()
                for idx, l in enumerate(len_seq):
                    position_info[idx][:l]=range(l,0,-1)
                
                
                loss, _ = sess.run([SASRec.loss, SASRec.opt],
                                   feed_dict={SASRec.item_seq: sample_seq,
                                              SASRec.len_seq: sp_len_seq,
                                              SASRec.behavior_seq: sample_beha,
                                              SASRec.next_behaviors_input: sample_target_beh,
                                              SASRec.position: position_info,
                                              SASRec.target: sample_target,
                                              SASRec.is_training:True})
               
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
            over_time_i = datetime.datetime.now()  
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
                                                                        have_dropout=True,
                                                                        is_test=True)

            if hit5 > best_hit_5 :
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'sasrec.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)
            else:
                count += 1
            if count == args.early_stop_epoch:
                break

    # with tf.Session() as sess :
    #     saver.restore(sess, 'sasrec.ckpt')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
                                                                # have_dropout=True, is_test=True)
    # #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
    # #                                                             have_dropout=True, have_user_emb=False,
    # #                                                             is_test=True,type='clicked')
    # #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
    # #                                                             have_dropout=True, have_user_emb=False,
    # #                                                             is_test=True,type='unclicked')







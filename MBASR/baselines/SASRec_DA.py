import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from utility import *
from SASRecModules import *
import random
import datetime
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
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
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
        self.is_training = tf.placeholder(tf.bool, shape=())

        all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

        self.input_emb=tf.nn.embedding_lookup(all_embeddings['item_embeddings'],self.item_seq)
        # Positional Encoding
        pos_emb=tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0), [tf.shape(self.item_seq)[0], 1]))
        self.seq=self.input_emb+pos_emb

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

        self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=tf.nn.softmax,scope='fc')

        self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(args.l2), tf.trainable_variables())
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss + self.reg)

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.emb_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.emb_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['pos_embeddings']=pos_embeddings
        return all_embeddings

def item_del(seq, behavior, gamma):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)
    sub_prob = cal_prob(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * gamma)
    num_samples = random.sample(range(num_samples+1), k=1)[0]
    if num_samples == 0:
        del_item_seq = np.concatenate(seq_)
        del_behavior_seq = np.concatenate(behavior_)
        return del_item_seq, del_behavior_seq
    else:
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            num_sampled_sub_seq = len(sub_seq)
             # 最后的购买item不能删除
            sub_seq_behavior = behavior_[index]
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_seq_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            if num_sampled_sub_seq_click == 0 or num_sampled_sub_seq == 1:
                pass
            else:
                del_index = random.sample(range(num_sampled_sub_seq_click), k=1)[0]
                deled_sub_seq = np.delete(sub_seq, del_index)
                deled_sub_behavior = np.delete(sub_seq_behavior, del_index)
                seq_[index] = deled_sub_seq
                behavior_[index] = deled_sub_behavior
                
        del_item_seq = np.concatenate(seq_)
        del_behavior_seq = np.concatenate(behavior_)
        # del_length = len(del_item_seq)

        return del_item_seq, del_behavior_seq
def item_reorder(seq, behavior, beta):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)
    sub_prob = cal_prob(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * beta)
    num_samples = random.sample(range(num_samples + 1), k=1)[0]
    if num_samples == 0:
        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)
        return reorder_item_seq, reorder_behavior_seq
    else:
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            sub_behavior = behavior_[index]
            num_sampled_sub_seq = len(sub_seq)
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            item_index = np.arange(num_sampled_sub_seq_click)
            if num_sampled_sub_seq_click == 0:
                continue
            item_prob = cal_prob(num_sampled_sub_seq_click)[::-1]
            if len(item_prob) == 1:
                continue
            indices = np.random.choice(item_index, p=item_prob, size=num_sampled_sub_seq_click, replace=False)
            if len(indices) == 1:
                continue
            shuffled_sub_seq = np.array(sub_seq)[indices].tolist() + sub_seq[num_sampled_sub_seq_click:]
            shuffled_sub_behavior = np.array(sub_behavior)[indices].tolist() + sub_behavior[num_sampled_sub_seq_click:]

            seq_[index] = shuffled_sub_seq
            behavior_[index] = shuffled_sub_behavior
            
        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)
        
        return reorder_item_seq, reorder_behavior_seq   

def sub_sequence_reorder(seq, alpha):
    seq_ = seq.copy()
    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)
    
    selected_item1_index = random.sample(range(num_sub_seq), k=1)[0]
    item_importance = np.power(alpha, abs(index - selected_item1_index))
    total = np.sum(item_importance)
    prob = item_importance / total
    selected_item2_index = np.random.choice(index, p=prob, size=1, replace=False)[0]
    
    seq_[selected_item1_index], seq_[selected_item2_index] = seq_[selected_item2_index], seq_[selected_item1_index]
    
    reorder_sub_seq = np.concatenate(seq_)
    
    return reorder_sub_seq

def augmentation(items, lengths, item_num, max_seq_len, tag, alpha, beta, gamma):
    batch_size = len(items)
    aug_items = []
    aug_lengths = []
    
    for i in range(batch_size):
        item_seq = items[i]
        length = lengths[i]
        if length <= 3:
            aug_items.append(item_seq)
            aug_lengths.append(length)
            continue
        sub_seq_len = random.sample(range(1, length//2), k=1)[0]
        unpad_item_seq = np.array(item_seq)[:length]
        
        item_sequences = [unpad_item_seq[start:start + sub_seq_len].tolist() for start in range(0, length - sub_seq_len + 1, sub_seq_len)]

        if tag == 1:
            aug_seq = item_del(item_sequences, gamma)
       
        elif tag == 2:
            aug_seq = sub_sequence_reorder(item_sequences, alpha)
        elif tag == 3:
            aug_seq = item_reorder(item_sequences, beta)
        
        elif tag == 0:
            aug_seq = unpad_item_seq
        
        item_seq = np.pad(aug_seq, (0, max_seq_len - len(aug_seq)), 'constant', constant_values=item_num)

        item_seq = item_seq.tolist()
        aug_items.append(item_seq)
        aug_lengths.append(len(aug_seq))
        
    return aug_items,aug_lengths

if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma = args.tag, args.alpha, args.beta, args.gamma

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the item_seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]

    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)


    SASRec = SASRecnetwork(hidden_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

    saver = tf.train.Saver(max_to_keep=10000)


    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'UB/model/da_sasrec/emb_{}_heads_{}_blocks_{}_dropout_{}_seed_{}_l2_{}_{}'.format(
        args.emb_size,args.num_heads,args.num_blocks,args.dropout_rate,args.random_seed,args.l2,nowTime)
    # save_dir = os.path.join(args.dataset,save_dir)
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
                len_seq = list(batch['len_seq'].values())
                
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                
                target=list(batch['target'].values())
                
                item_seq, len_seq = augmentation(item_seq, len_seq, item_num, state_size,  tag, alpha, beta, gamma)
                 
                loss, _ = sess.run([SASRec.loss, SASRec.opt],
                                   feed_dict={SASRec.item_seq: item_seq,
                                              SASRec.len_seq: len_seq,
                                              SASRec.target: target,
                                              SASRec.is_training:True})
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
            over_time_i = datetime.datetime.now()  # 程序结束时间
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, SASRec, data_directory, topk,
                                                                        have_dropout=True,have_user_emb=False,
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
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, SASRec, data_directory, topk,
    #                                                                 have_dropout=True, have_user_emb=False,
    #                                                                 is_test=True)
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, SASRec, data_directory, topk,
    #                                                                 have_dropout=True, have_user_emb=False,
    #                                                                 is_test=True,type='clicked')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, SASRec, data_directory, topk,
    #                                                                 have_dropout=True, have_user_emb=False,
    #                                                                 is_test=True,type='unclicked')








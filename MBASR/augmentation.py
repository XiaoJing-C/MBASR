import pandas as pd
import numpy as np
import random
import math


def cal_prob(length, a=0.8):
    item_indices = np.arange(length)  # 创建从 0 到 n-1 的索引张量
    item_importance = np.power(a, length - item_indices)

    total = np.sum(item_importance)
    prob = item_importance / total
    return prob
    
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

 
def sub_sequence_reorder(seq, behavior, alpha):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)
    
    selected_item1_index = random.sample(range(num_sub_seq), k=1)[0]
    item_importance = np.power(alpha, abs(index - selected_item1_index))
    total = np.sum(item_importance)
    prob = item_importance / total
    selected_item2_index = np.random.choice(index, p=prob, size=1, replace=False)[0]
    
    seq_[selected_item1_index], seq_[selected_item2_index] = seq_[selected_item2_index], seq_[selected_item1_index]
    behavior_[selected_item1_index], behavior_[selected_item2_index] = behavior_[selected_item2_index], behavior_[selected_item1_index]
    
    reorder_sub_seq = np.concatenate(seq_)
    reorder_sub_behavior = np.concatenate(behavior_)
    
    return reorder_sub_seq, reorder_sub_behavior

 
def augmentation(items, behavios, lengths, item_num, max_seq_len, tag, alpha, beta, gamma, lamda):
    batch_size = len(items)
    aug_items = []
    aug_behaviors = []
    aug_lengths = []
    for i in range(batch_size):
        item_seq = items[i]
        behavior_seq = behavios[i]
        length = lengths[i]
        
        unpad_item_seq = np.array(item_seq)[:length]
        unpad_behavior_seq = np.array(behavior_seq)[:length]
        
        mask = (unpad_behavior_seq[:-1] == 1) & (unpad_behavior_seq[1:] == 0)
        
        split_indices = np.where(mask)[0] + 1
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, length)
        item_sequences = [unpad_item_seq[start:end].tolist() for start, end in zip(split_indices[:-1], split_indices[1:])]
        behavior_sequences = [unpad_behavior_seq[start:end].tolist() for start, end in zip(split_indices[:-1], split_indices[1:])]
        
        if tag == 1:
            aug_seq, aug_behavior = item_del(item_sequences, behavior_sequences, gamma)
        elif tag == 2:
            aug_seq, aug_behavior = sub_sequence_reorder(item_sequences, behavior_sequences, alpha)
        elif tag == 3:
            aug_seq, aug_behavior = item_reorder(item_sequences, behavior_sequences, beta)

        elif tag == 0:
            aug_seq, aug_behavior = unpad_item_seq, unpad_behavior_seq
        
        item_seq = np.pad(aug_seq, (0, max_seq_len - len(aug_seq)), 'constant', constant_values=item_num)
        behavior_seq = np.pad(aug_behavior, (0, max_seq_len - len(aug_behavior)), 'constant', constant_values=2)
        
        item_seq = item_seq.tolist()
        behavior_seq = behavior_seq.tolist()
        
        aug_items.append(item_seq)
        aug_behaviors.append(behavior_seq)
        aug_lengths.append(len(aug_seq))
        
    return aug_items, aug_behaviors, aug_lengths
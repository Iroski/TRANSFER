import os
import javalang
import pickle
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import string
import random
import sys


def solve_camel_and_underline(token):
    if token.isdigit():
        return [token]
    else:
        p = re.compile(r'([a-z]|\d)([A-Z])')
        sub = re.sub(p, r'\1_\2', token).lower()
        sub_tokens = sub.split("_")
        tokens = re.sub(" +", " ", " ".join(sub_tokens)).strip()
        final_token = []
        for factor in tokens.split(" "):
            final_token.append(factor.rstrip(string.digits))
        return final_token


def cut_data(token_seq, token_length_for_reserve):
    if len(token_seq) <= token_length_for_reserve:
        return token_seq
    else:
        start_index = token_seq.index("rank2fixstart")
        end_index = token_seq.index("rank2fixend")
        assert end_index > start_index
        length_of_annotated_statement = end_index - start_index + 1
        if length_of_annotated_statement <= token_length_for_reserve:
            padding_length = token_length_for_reserve - length_of_annotated_statement
            # give 2/3 padding space to content before annotated statement
            pre_padding_length = padding_length // 3 * 2
            # give 1/3 padding space to content after annotated statement
            post_padding_length = padding_length - pre_padding_length
            if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
                return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
            elif start_index < pre_padding_length:
                return token_seq[:token_length_for_reserve]
            elif len(token_seq) - end_index - 1 < post_padding_length:
                return token_seq[len(token_seq) - token_length_for_reserve:]
        else:
            return token_seq[start_index: start_index + token_length_for_reserve]


def generate_token_seq(tokens,token_length_for_reserve):
	last_token=None
	t_seq = []
	for token in tokens:
		if isinstance(token, javalang.tokenizer.String):
			tmp_token = ["stringliteral"]
		else:
			tmp_token = solve_camel_and_underline(token.value)
			if last_token is not None and isinstance(token, javalang.tokenizer.Identifier) and (
					isinstance(last_token, javalang.tokenizer.Identifier) or isinstance(last_token,
																						   javalang.tokenizer.BasicType)) and last_token.value!='rank2fixstart':
				cur_token=tmp_token
				tmp_token=[]
				type_token_seq = solve_camel_and_underline(last_token.value)
				for t_token in cur_token:
					tmp_token.append([type_token_seq, cur_token[:cur_token.index(t_token)], t_token])
		last_token=token
		t_seq += tmp_token
	return cut_data(t_seq, token_length_for_reserve)

def token2num(tokens,vocab_dict,index_oov):
    num_seq=[]
    for token in tokens:
        if type(token)==str:
            num=vocab_dict[token] if token in vocab_dict else index_oov
        else:

            identifiers=token[0]
            print(identifiers)
            prev_token=token[1]
            cur_token=token[2]
            iden_nums=[vocab_dict[t] if t in vocab_dict else index_oov for t in identifiers]
            prev_nums=[vocab_dict[t] if t in vocab_dict else index_oov for t in prev_token]
            cur_num=vocab_dict[cur_token] if cur_token in vocab_dict else index_oov
            num=[iden_nums,prev_nums,cur_num]
        num_seq.append(num)
    return num_seq


if __name__ == "__main__":

    current_pattern = sys.argv[1]
    pattern_list = ["InsertMissedStmt",
                    "InsertNullPointerChecker",
                    "MoveStmt",
                    "MutateConditionalExpr",
                    "MutateDataType",
                    "MutateLiteralExpr",
                    "MutateMethodInvExpr",
                    "MutateOperators",
                    "MutateReturnStmt",
                    "MutateVariable",
                    "RemoveBuggyStmt"]
    if current_pattern not in pattern_list:
        print("The fix pattern specified by the argument dost not exist.")
        exit(-1)

    print("Current fix pattern: {}".format(current_pattern))
    # Path declaration
    print("Path declaration")
    fl_dataset_path = "../../dataset_fl.pkl"
    output_data_dir = "../data/{}/".format(current_pattern)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # Parameters declaration
    print("Parameters declaration")
    tags = ["positive", "negative"]
    train_prop = 0.8
    val_prop = 0.1
    test_prop = 0.1
    shuffle_seed = 666
    vector_size = 32
    token_length_for_reserve = 400
    extend_size = 2
    max_vocab_size = 50000

    # split train/val/test parts and generate training corpus for word2vec pre-training
    with open(fl_dataset_path, "rb") as file:
        dataset_fl = pickle.load(file)
    token_seq_dataset = {"train": {}, "val": {}, "test": {}}
    w2v_training_corpus = []
    for tag in dataset_fl:
        for part in token_seq_dataset:
            token_seq_dataset[part][tag] = []
        random_copy = dataset_fl[tag][current_pattern][:]
        random.seed(shuffle_seed)
        random.shuffle(random_copy)
        for index, method in enumerate(random_copy):
            method = method.strip()
            tokens = javalang.tokenizer.tokenize(method)

            token_seq = generate_token_seq(tokens,token_length_for_reserve)

            if index + 1 < len(random_copy) * train_prop:
                token_seq_dataset["train"][tag].append(token_seq)
            elif index + 1 < len(random_copy) * (train_prop + val_prop):
                token_seq_dataset["val"][tag].append(token_seq)
            else:
                token_seq_dataset["test"][tag].append(token_seq)


    print("load vocab and vectors")
    with open(os.path.join(output_data_dir, "vocab.pkl"), "rb") as file:
        vocab_dict=pickle.load(file)
    with open(os.path.join(output_data_dir, "vectors.pkl"), "rb") as file:
        vectors= pickle.load(file)

    # Generating train/val/test dataset
    print("Generating train/val/test dataset")
    index_oov = vocab_dict["OOV"]
    index_padding = vocab_dict["PADDING"]
    current_index = len(vocab_dict)

    for part in token_seq_dataset:
        normal_corpus = []
        normal_labels = []
        for tag in token_seq_dataset[part]:
            for token_seq in token_seq_dataset[part][tag]:
                normal_record = token2num(token_seq,vocab_dict,index_oov)
                if len(normal_record) < token_length_for_reserve:
                    normal_record += [index_padding] * (token_length_for_reserve - len(normal_record))
                normal_corpus.append(normal_record)
            normal_labels += [tags.index(tag)] * len(token_seq_dataset[part][tag])
        random.seed(shuffle_seed)
        random.shuffle(normal_corpus)
        random.seed(shuffle_seed)
        random.shuffle(normal_labels)
        if not os.path.exists(os.path.join(output_data_dir, part)):
            os.makedirs(os.path.join(output_data_dir, part))
        with open(os.path.join(output_data_dir, part, "x_w2v_new.pkl"), "wb") as file:
            pickle.dump(normal_corpus, file)
        with open(os.path.join(output_data_dir, part, "y_.pkl"), "wb") as file:
            pickle.dump(normal_labels, file)

    print("Done")

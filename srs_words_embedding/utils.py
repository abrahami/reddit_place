import random


def sentences_yielder(dp_obj, sr_obj, config_dict, verbose=True):
    config_dict_filters = config_dict["text_filters"]
    subm_and_comments = []
    for st in sr_obj.submissions_as_list:
        if type(st[2]) is str and type(st[1]) is str:
            subm_and_comments.append((st[0], st[1] + ' . ' + st[2], st[5]))
        elif type(st[1]) is str:
            subm_and_comments.append((st[0], st[1], st[5]))
        elif type(st[2]) is str:
            subm_and_comments.append((st[0], st[2], st[5]))
        else:
            continue
    for st in sr_obj.comments_as_list:
        if type(st[1]) is str:
            subm_and_comments.append((st[0], st[1], st[5]))
        else:
            continue

    # first, in case we wish to filter data based on the score of each submission/comment
    if eval(config_dict_filters["sampling_rules"]["sample_data"]):
        subm_and_comments = list(filter(lambda tup: abs(tup[0]) >=
                                                    config_dict_filters["sampling_rules"]["min_abs_score"],
                                        subm_and_comments))
    # secondly, sorting the submissions/comments according to the logic required
    if config_dict_filters["ordering_logic"] == 'score':
        subm_and_comments.sort(key=lambda tup: tup[0], reverse=True)
    # case we want to sort the data by date - not an issue
    elif config_dict_filters["ordering_logic"] == 'date':
        subm_and_comments.sort(key=lambda tup: tup[2], reverse=True)
    # case we want to randomly sort the submissions
    elif config_dict_filters["ordering_logic"] == 'random':
        random.seed(config_dict["random_seed"])
        random.shuffle(subm_and_comments)
        random.shuffle(subm_and_comments)
    # lastly, if we wish to sample only part of the data
    if eval(config_dict_filters["sampling_rules"]["sample_data"]) and \
            config_dict_filters["sampling_rules"]["max_posts"] is not None:
        subm_and_comments = subm_and_comments[0:config_dict_filters["sampling_rules"]["max_posts"]]

    full_tok_text = []
    for st in subm_and_comments:
        normalized_text = dp_obj.mark_urls(st[1], marking_method='tag')[0]
        cur_tok_words = dp_obj.tokenize_text(sample=normalized_text, convert_to_lemmas=False, break_to_sents=True)
        full_tok_text.extend(cur_tok_words)
    if verbose:
        print("Finished handling sr {}. {} submissions/comments were processed, "
              "yielded {} sentences".format(sr_obj.name, len(subm_and_comments), len(full_tok_text)))
    return full_tok_text

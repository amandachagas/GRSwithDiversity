from GRSD import GRSD
import constants
import utils


def flow(grsd, technique = 'AWM'):
    print("\n-->  Initializing...")
    grsd.set_k()


    my_group = grsd.random_group(5)
    print('\n-->  Group members: {}'.format(my_group))


    grsd.predict_ratings(group=my_group)


    grsd.set_profile_items(group=my_group)
    grsd.set_candidate_items()


    print("\n\n-->  Calculating items similarity matrix...")
    grsd.calc_similarity_matrix()


    print("\n\n-->  Calculating group matrix FILLED...")
    group_filled_mtx = grsd.group_sparse_mtx.copy()

    for index, row in group_filled_mtx.iterrows():
        for col in list(group_filled_mtx):
            if(group_filled_mtx.loc[index,col] == 0.0):
                aux = list(filter(lambda x: x.uid==str(index) and x.iid==str(col), grsd.predictions))
                group_filled_mtx.loc[index,col] = aux[0].est

    group_filled_mtx = group_filled_mtx.round(decimals=3)

    print("\n\n-->  Applying aggregation technique...")
    agg_group_profile = utils.apply_aggregation_strategy(group_filled_mtx, technique)


    print("\n\n-->  Creating group preferences dict...")
    group_pref_dict = []
    for col in list(agg_group_profile):
        my_dict = {}
        my_dict['rating'] = agg_group_profile.loc[900,col]
        my_dict['movieID'] = col
        group_pref_dict.append(my_dict)
        
    group_pref_dict = sorted(group_pref_dict, key = lambda i: i['rating'],reverse=True)


    references = group_pref_dict[0:10]


    print("\n\n-->  Calculating recommendations...")
    recs = grsd.get_similar_items(references)
    candidates_list = grsd.get_relevance_score(recs=recs, references=references)
    

    print("\n\n-->  The top-20 STANDARD recs are:\n")
    for item in candidates_list[0:20]:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))



    my_candidates = candidates_list.copy()
    final_recs_greedy = grsd.diversify_recs_list(recs=my_candidates)
    print("\n\n-->  The top-10 GREEDY DIVERSIFIED recs are:\n")
    for item in final_recs_greedy:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))



    my_candidates = candidates_list.copy()
    final_recs_random = grsd.diversify_recs_list_bounded_random(recs=my_candidates)
    print("\n\n-->  The top-10 RANDOM DIVERSIFIED recs are:\n")
    for item in final_recs_random:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))


    print('\n\n')
    print("########################################################################")
    print("#######################     EVALUATING SYSTEM    #######################")
    print("########################################################################")
    print('\n\n')

    standard_recs = candidates_list[0:10]

    ild_s = grsd.get_ILD_score(standard_recs, title_weight=0.8)
    ild_g = grsd.get_ILD_score(final_recs_greedy, title_weight=0.8)
    ild_r = grsd.get_ILD_score(final_recs_random, title_weight=0.8)
    p3_s = grsd.precision_at(standard_recs, 3)
    p3_g = grsd.precision_at(final_recs_greedy, 3)
    p3_r = grsd.precision_at(final_recs_random, 3)
    p5_s = grsd.precision_at(standard_recs, 5)
    p5_g = grsd.precision_at(final_recs_greedy, 5)
    p5_r = grsd.precision_at(final_recs_random, 5)
    p10_s = grsd.precision_at(standard_recs, 10)
    p10_g = grsd.precision_at(final_recs_greedy, 10)
    p10_r = grsd.precision_at(final_recs_random, 10)

    p_3_5_10_s = [p3_s, p5_s, p10_s]  
    p_3_5_10_g = [p3_g, p5_g, p10_g]
    p_3_5_10_r = [p3_r, p5_r, p10_r]

    evaluation = dict()
    evaluation['ild_s'] = ild_s
    evaluation['ild_g'] = ild_g
    evaluation['ild_r'] = ild_r
    evaluation['p_3_5_10_s'] = p_3_5_10_s
    evaluation['p_3_5_10_g'] = p_3_5_10_g
    evaluation['p_3_5_10_r'] = p_3_5_10_r

    total_recs = dict()
    total_recs['recs_standard'] = standard_recs
    total_recs['recs_greedy'] = final_recs_greedy
    total_recs['recs_random'] = final_recs_random



    print('ILD - standard recs: {}'.format(ild_s))
    print('ILD - div greedy algo: {}'.format(ild_g))
    print('ILD - div random algo: {}'.format(ild_r))
    print('\n')
    print('P@3 - standard recs: {}\n'.format(p3_s))
    print('P@5 - standard recs: {}\n'.format(p5_s))
    print('P@10 - standard recs: {}\n'.format(p10_s))
    print('\n')
    print('\n')
    print('P@3 - div greedy algo: {}\n'.format(p3_g))
    print('P@5 - div greedy algo: {}\n'.format(p5_g))
    print('P@10 - div greedy algo: {}\n'.format(p10_g))
    print('\n')
    print('\n')
    print('P@3 - div random algo: {}'.format(p3_r))
    print('P@5 - div random algo: {}'.format(p5_r))
    print('P@10 - div random algo: {}'.format(p10_r))

    return total_recs, evaluation


grsd = GRSD(rating_data=constants.RATINGS_PATH, item_data=constants.ITEMS_PATH)
divRecs, evaluation = flow(grsd, technique = 'AWM')

print('\n\n')
print("########################################################################")
print("########################        DONE       #############################")
print("########################################################################")
print('\n\n')
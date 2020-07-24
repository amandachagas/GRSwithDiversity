import pandas as pd

def apply_aggregation_strategy(group_filled_mtx, technique = 'AWM'):
    ''' Sets the aggregation technique applied.
        Returns the group profile aggregated.
    '''
    values = []
    labels = []
    for i in range(0,len(list(group_filled_mtx))):
        my_col = group_filled_mtx.iloc[ : ,i]
        label = my_col.name
        my_col = list(my_col)

        labels.append(label)
        values.append(0.0)
        
        
        if technique is 'LM':
            values.append( float(min(my_col)) )
        elif technique is 'MP':
            values.append( float(max(my_col)) )
        else:
            if float(min(my_col)) <= 2 :
                values.append( float(min(my_col)) )
            else:
                values.append( float( sum(my_col) / len(my_col) ) )
                

    print('\n-- -- --  -- > Aggregation Technique chosen: {}\n'.format(technique))
    
    # print('Array values: {}, Array labels: {}'.format(values, labels))
    agg_group_profile = pd.DataFrame(index=[900], columns=labels)

    for i in range(0,len(list(agg_group_profile))):
        agg_group_profile.iloc[0, i] = values[i]

    agg_group_profile = agg_group_profile.round(decimals=3)
    
    return agg_group_profile
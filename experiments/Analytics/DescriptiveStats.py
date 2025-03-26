#!/usr/bin/env python3
#
# DataManipulation.py

class DataManipulation:
    def __init__(self, data):
        self.data = data

    def merge_datasets(self, second_data, left_on, right_on):
        try:
            merged_data = pd.merge(left=self.data, right=second_data, left_on=left_on, right_on=right_on, how='inner')
            return merged_data
        except Exception as e:
            print("Error while merging datasets:", e)

    def join_datasets(self, second_data, left_on, right_on, how='inner'):
        try:
            joined_data = pd.join(self.data, second_data, lsuffix='_left', rsuffix='_right', on=[left_on, right_on], how=how)
            return joined_data
        except Exception as e:
            print("Error while joining datasets:", e)

    def concatenate_datasets(self, *args, ignore_index=True):
        try:
            concatenated_data = pd.concat([self.data] + list(args), ignore_index=ignore_index)
            return concatenated_data
        except Exception as e:
            print("Error while concatenating datasets:", e)

    def drop_duplicate_rows(self, subset=None, keep=True, inplace=True):
        try:
            self.data.drop_duplicates(subset=subset, keep=keep, inplace=inplace)
            return self.data
        except Exception as e:
            print("Error while dropping duplicate rows:", e)

    def fill_na_with_constant(self, constant):
        try:
            filled_data = self.data.fillna(constant)
            return filled_data
        except Exception as e:
            print("Error while filling NaNs with constants:", e)

    def aggregate_by_group(self, group_cols, aggr_func, new_col_name):
        try:
            aggr_data = self.data.groupby(group_cols)[aggr_func].transform(lambda x: x.mean()).rename(new_col_name)
            self.data = self.data.assign(**{new_col_name: aggr_data})
            return self.data
        except Exception as e:
            print("Error while aggregating by group:", e)

    def filter_data(self, condition):
        try:
            filtered_data = self.data[condition]
            return filtered_data
        except Exception as e:
            print("Error while filtering data:", e)

    def replace_values(self, old_val, new_val):
        try:
            replaced_data = self.data.replace({old_val: new_val})
            return replaced_data
        except Exception as e:
            print("Error while replacing values:", e)

import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        data = []

        schema = (
            "tables: flight(flight_id,from_airport,to_airport,airline_code,departure_time,arrival_time,flight_days,stops), "
            "airport(airport_code,airport_name,state_code), "
            "airport_service(city_code,airport_code), "
            "city(city_code,city_name,state_code), "
            "airline(airline_code,airline_name), "
            "fare(fare_id,from_airport,to_airport,fare_airline,one_direction_cost,round_trip_cost), "
            "fare_basis(fare_basis_code,class_type,economy,discounted,night), "
            "flight_fare(flight_id,fare_id), "
            "days(days_code,day_name), "
            "date_day(month_number,day_number,year,day_name), "
            "time_interval(period,begin_time,end_time), "
            "ground_service(city_code,airport_code,transport_type), "
            "aircraft(aircraft_code,aircraft_description,manufacturer), "
            "state(state_code,state_name), "
            "restriction(restriction_code,stopovers,minimum_stay,maximum_stay)"
        )

        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)

            for nl, sql in zip(nl_lines, sql_lines):
                encoder_ids = tokenizer.encode(
                    "translate English to SQL: " + schema + " | " + nl,
                    add_special_tokens=True
                )

                sql_ids = tokenizer.encode(sql, add_special_tokens=False)
                eos_id = tokenizer.eos_token_id

                bos_id = tokenizer.pad_token_id

                decoder_input = [bos_id] + sql_ids
                decoder_targets = sql_ids + [eos_id]

                data.append({
                    'encoder_ids': encoder_ids,
                    'decoder_input': decoder_input,
                    'decoder_targets': decoder_targets,
                    'initial_decoder_input': [bos_id],
                })
        else:
            for nl in nl_lines:
                encoder_ids = tokenizer.encode(
                    "translate English to SQL: " + schema + " | " + nl,
                    add_special_tokens=True
                )
                bos_id = tokenizer.pad_token_id

                data.append({
                    'encoder_ids': encoder_ids,
                    'initial_decoder_input': [bos_id],
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    encoder_ids_list = [torch.tensor(item['encoder_ids'], dtype=torch.long) for item in batch]
    decoder_input_list = [torch.tensor(item['decoder_input'], dtype=torch.long) for item in batch]
    decoder_targets_list = [torch.tensor(item['decoder_targets'], dtype=torch.long) for item in batch]

    initial_decoder_inputs = torch.tensor(
        [[item['initial_decoder_input'][0]] for item in batch], dtype=torch.long
    )

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    encoder_ids_list = [torch.tensor(item['encoder_ids'], dtype=torch.long) for item in batch]

    initial_decoder_inputs = torch.tensor(
        [[item['initial_decoder_input'][0]] for item in batch], dtype=torch.long
    )

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x

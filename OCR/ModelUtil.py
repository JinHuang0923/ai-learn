import torch
import torchinfo
from copy import deepcopy
from torch.nn import Module
import random
from typing import Dict, List

class ModelSummary:
    def __init__(self, module_name,module_type,param_num,sparsity):
        self.module_name = module_name
        self.module_type = module_type
        self.param_num = param_num
        self.sparsity = sparsity
    def __repr__(self):
        return f"module_name: {self.module_name}, module_type: {self.module_type}, param_num: {self.param_num}, sparsity: {self.sparsity}"
def get_model_summary(model):
    pass
def compute_sparsity_compact2origin(origin_model: Module, compact_model: Module, config_list: List[Dict]) -> List[Dict]:
    """
    Compare origin model and compact model, return the sparsity of each group mentioned in config list.
    A group means all layer mentioned in one config.
    e.g., a linear named 'linear1' and its weight size is [100, 100] in origin model, but in compact model,
    the layer weight size with same layer name is [100, 50],
    then this function will return [{'op_names': 'linear1', 'total_sparsity': 0.5}].
    """
    compact2origin_sparsity = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for module_name, module in origin_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            total_weight_num += module.weight.data.numel()  # type: ignore
        for module_name, module in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            left_weight_num += module.weight.data.numel()  # type: ignore
        compact2origin_sparsity.append(deepcopy(config))
        compact2origin_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return compact2origin_sparsity
def numel(model):
    return sum([p.numel() for p in model.parameters()])
def expect_numel(model_summary_list):
    total_param = 0
    for key,p in model_summary_list.items():
        if p.param_num == -1:
            continue
        total_param += (p.param_num*p.sparsity)
    return total_param

def filter_prun_dict(summary_dict):
    prun_dict = {}
    for key, value in summary_dict.items():
        if value.module_type == 'Conv2d' or value.module_type == 'Linear':
            prun_dict[key] = value
    return prun_dict

def pick_random_key_from_dict(d: dict):
    """Grab a random key from a dictionary."""
    keys = list(d.keys())
    random_key = random.choice(keys)
    return random_key

# def pick_random_item_from_dict(d: dict):
#     """Grab a random item from a dictionary."""
#     random_key = pick_random_key_from_dict(d)
#     random_item = random_key, d[random_key]
#     return random_item
def get_min_sparsity_item(compute_dict):
    min_sparsity_key = None
    min_sparsity = 1
    for key, value in compute_dict.items():
        if value.sparsity < min_sparsity:
            min_sparsity_key = key
            min_sparsity = value.sparsity
    return min_sparsity_key
def convert_config_list(computed_dict):
    config_list = []
    for _,item in computed_dict.items():
        if item.sparsity < 1.0:
            config_item = {
                "total_sparsity": 1 - item.sparsity,
                "op_names": [item.module_name]
            }
            config_list.append(config_item)
    return config_list

def generate_compute_dict(model_summary_list,expected_sparsity):
    expect_param_num = expect_numel(model_summary_list) * expected_sparsity
    prun_dict = filter_prun_dict(model_summary_list)

    # example: sparsity:0,7
    compute_dict = deepcopy(model_summary_list)


    while(True):
        current_compute_dict_param_num = expect_numel(compute_dict)

        print(f"current compute dict param num : {current_compute_dict_param_num}")
        # 如果预估参数量 > 期望模型总参数量,则减小稀疏度(减小参数量)
        if current_compute_dict_param_num > expect_param_num:
            prun_dict,compute_dict = decrease_sparsity(prun_dict=prun_dict,compute_dict=compute_dict)
            if prun_dict == None:
                return None
        else:
        # 预估参数量 < 期望模型总参数量
            if current_compute_dict_param_num >= expect_param_num * 0.9:
                # 在可接受范围内,则返回可用
                return compute_dict
            else:
                # 预估参数量小的太多了,再加点稀疏度
                compute_dict = increase_compute_dict_sparsity(compute_dict=compute_dict)


def decrease_sparsity(prun_dict,compute_dict):
    if len(prun_dict) == 0:
        # 已经全部随机了一遍还未满足期望要求稀疏度,则重新随机
        return None,None
    # randomly get item from prun_dict
    random_key = pick_random_key_from_dict(prun_dict)
    # remove this key from prun_dict
    del prun_dict[random_key]
    # TODO 这里应该根据稀疏度决定区间
    random_sparsity = random.randint(35, 99) / 100
    compute_dict[random_key].sparsity = random_sparsity

    # print(f"decrease_sparsity,name: {random_key} sparsity: {random_sparsity}")
    # set the item corresponding sparsity
    return prun_dict, compute_dict


def increase_compute_dict_sparsity(compute_dict):
    min_item_key = get_min_sparsity_item(compute_dict)
    min_item_sparsity = compute_dict[min_item_key].sparsity
    # 随机增加稀疏度(当前稀疏度~0.99)
    new_random_sparsity = random.randint(int(min_item_sparsity * 100), 99) / 100
    # 修改放回计算dict
    compute_dict[min_item_key].sparsity = new_random_sparsity
    # print(f"increase_compute_dict_sparsity,name: {min_item_key} sparsity: {new_random_sparsity}")
    return compute_dict

def prun_model_layer(model):
    model_net_list = {}
    # total_param = 0
    for module_name, module in model.named_modules():
            # print(module_name)
            # print(type(module))
            module_type = type(module).__name__
            param_numdel = -1
            try:
                param_numdel = module.weight.data.numel()
            except AttributeError:
                pass
            # param_numdel = module.weight.data.numel()
            model_prun_layer = ModelSummary(module_name, module_type, param_numdel, 1)
            model_net_list[module_name] = model_prun_layer
    # print(f"total_param:{total_param}")
    return model_net_list
def generate_config_list(model,expected_sparsity):
    model_summary_list = prun_model_layer(model)
    total_param_num = expect_numel(model_summary_list)
    print(f"total_param_num: {total_param_num}")
    compute_dict = None
    count = 0
    while(compute_dict == None):
        compute_dict = generate_compute_dict(model_summary_list,expected_sparsity)
        if compute_dict == None:
            count += 1
            print(f"RETRY!!!! count: {count}")

    print(f"finish compute num: {expect_numel(compute_dict)}")
    # print(compute_dict)

    config_list = convert_config_list(compute_dict)
    return config_list


if __name__ == '__main__':
    print('Start')
    # model = torch.load("./2023-01-09-16-12-22-847212/best_result/model.pth");
    # model = VGG().to(device)
    model = torch.load("../../../model-repo/idp_CIFAR10_model.pth").to("cuda")
    # print(type(mode))
    model_summary_list = prun_model_layer(model)
    total_param_num = expect_numel(model_summary_list)
    print(f"total_param_num: {total_param_num}")
    prun_dict = filter_prun_dict(model_summary_list)

    compute_dict = generate_compute_dict(prun_dict)

    print(f"finish compute num: {expect_numel(compute_dict)}")
    print(compute_dict)

    config_list = convert_config_list(compute_dict)
    print(config_list)


    # # print(model.__dict__)
    # param_numel = numel(model)

    # print(param_numel)
    # module_class = torchinfo.summary(model)
    # list = module_class.summary_list
    # print(module_class)


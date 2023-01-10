import os
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_path", type=str)
arg_parser.add_argument("--use-discretized-grad", action='store_true')
arg_parser.add_argument("--discretized-grad-renew", action='store_true')
arg_parser.add_argument("--stochastic-rounding", action='store_true')
arg_parser.add_argument("--for-speed", action='store_true')
arg_parser.add_argument("--device", type=str, default='cpu')
arg_parser.add_argument("--force-col-wise", action='store_true')

script_fname = 'run.sh'
running = open(script_fname, 'w')
os.system(f"chmod +x {script_fname}")

data = [
    'higgs',
    'epsilon',
    'criteo',
    'bosch',
    'kitsune',
    'yahoo',
    'msltr',
    'year'
]

task = [
    'binary',
    'binary',
    'binary',
    'binary',
    'binary',
    'ranking',
    'ranking',
    'regression'
]

col_wise_data = ['epsilon', 'year', 'yahoo', 'bosch']

bins = [2, 3, 4, 5]

def generate_script(data_path, use_discretized_grad, discretized_grad_renew, stochastic_rounding, for_speed, device):

    dataset = [
        f'data={data_path}/higgs.train',
        f'data={data_path}/epsilon.train',
        f'data={data_path}/criteo.train',
        f'data={data_path}/bosch.train',
        f'data={data_path}/kitsune.train',
        f'data={data_path}/yahoo.train',
        f'data={data_path}/msltr.train',
        f'data={data_path}/year.train',
    ]

    validset = [
        f'valid={data_path}/higgs.test',
        f'valid={data_path}/epsilon.test',
        f'valid={data_path}/criteo.test',
        f'valid={data_path}/bosch.test',
        f'valid={data_path}/kitsune.test',
        f'valid={data_path}/yahoo.test',
        f'valid={data_path}/msltr.test',
        f'valid={data_path}/year.test'
    ]

    os.system("mkdir -p logs")
    use_discretized_grad_str = str(use_discretized_grad).lower()
    discretized_grad_renew_str = str(discretized_grad_renew).lower()
    stochastic_rounding_str = str(stochastic_rounding).lower()
    num_k = 4 if use_discretized_grad else 1
    for i in range(8):
        for j in range(5):
            for k in range(num_k):
                base_conf_fname = 'train_model.conf' if task[i] == 'binary' else ('train_rank_model.conf' if task[i] == 'ranking' else 'train_reg_model.conf')
                args = ''
                args += dataset[i]
                if not for_speed:
                    args += ' ' + validset[i]
                args += ' seed=' + str(j)
                if use_discretized_grad:
                    args += ' grad_discretize_bins='+str(2**bins[k]-2)
                    log_name = './logs/train_' + data[i] + '_seed'+str(j) + '_bins' + str(bins[k])+'.log'
                else:
                    log_name = './logs/train_' + data[i] + '_seed'+str(j)+ '_fp32' + '.log'
                args += f' use_discretized_grad={use_discretized_grad_str} discretized_grad_renew={discretized_grad_renew_str} stochastic_rounding={stochastic_rounding_str}'
                if data[i] == 'bosch':
                    args += ' learning_rate=0.015'
                if data[i] in col_wise_data:
                    args += ' force_row_wise=false force_col_wise=true'
                if device != 'cpu':
                    args += f' device_type={device}'
                running.write(f'../LightGBM/lightgbm config={base_conf_fname} {args} > {log_name} 2>&1\n')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    generate_script(args.data_path, args.use_discretized_grad, args.discretized_grad_renew, args.stochastic_rounding, args.for_speed, args.device)

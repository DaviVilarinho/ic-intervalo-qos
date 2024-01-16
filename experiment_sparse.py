from subprocess import call

if __name__ == '__main__':
    call_files = [
        'experiment_replication_x_switch.py'
        'experiment_replication_x_univariate.py'
        'experiment_replication_x_flow.py'
        'experiment_replication_x_port.py'
        'experiment_replication_x_total_kv.py'
        'experiment_replication_x_total_vod.py'
        'experiment_replication_x_stepwise.py'
    ]

    for call_file in call_files:
        for strategy in ['Naive']:
            for sparsing_factor in list(range(600,0, -15)) + [1]:
                call(['python3', call_file, '--strategy', strategy, '--sparsing-factor', str(sparsing_factor), '--destination', 'sparsing'])


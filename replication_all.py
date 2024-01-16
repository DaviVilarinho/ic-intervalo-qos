from subprocess import call
import concurrent.futures


def process_file(call_file):
    call(['python3', call_file, '--sparsing-factor', '1'])


if __name__ == '__main__':
    call_files = [
        'experiment_replication_x_switch.py',
        'experiment_replication_x_univariate.py',
        'experiment_replication_x_flow.py',
        'experiment_replication_x_port.py',
        'experiment_replication_x_total_kv.py',
        'experiment_replication_x_total_vod.py'
        #        'experiment_replication_x_stepwise.py'
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit the tasks to the executor
        futures = [executor.submit(process_file, call_file)
                   for call_file in call_files]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

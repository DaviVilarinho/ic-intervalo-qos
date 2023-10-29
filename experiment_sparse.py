from subprocess import call

if __name__ == '__main__':
    call(['python3', ''])
    call(['python3', 'experiment_on_correlation.py'])
    call(['python3', 'experiment_replication_switch.py'])
    call(['python3', 'experiment_replication_x_flow.py'])
    call(['python3', 'experiment_replication_x_port.py'])
    call(['python3', 'experiment_replication_x_total_kv.py'])
    call(['python3', 'experiment_replication_x_total_vod.py'])


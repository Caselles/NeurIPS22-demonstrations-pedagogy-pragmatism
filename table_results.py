import numpy as np
import os

path_table = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/job_SIGNIFICANT_goal_pred_sqil_demos7_exp3_500_demos/'

final_results = {}

for config in ['naive_literal', 'naive_pragmatic', 'pedagogical_literal', 'pedagogical_pragmatic']:

    print(config)

    path_config = path_table + config + '/'

    dirs = os.listdir(path_config)

    prs = []
    srs = []

    for d in dirs:
        with open(path_config+d+'/results_predictability_reachability.txt') as f:
            lines = f.readlines()

        prs.append(float(lines[0].split(':')[1].split('\n')[0]))
        srs.append(float(lines[1].split(':')[1].split('\n')[0]))


    m_prs = np.mean(prs)
    m_srs = np.mean(srs)

    std_prs = np.std(prs)
    std_srs = np.std(srs)

    print(m_prs, m_srs)
    print(std_prs, std_srs)
    print(m_prs-std_prs, m_prs+std_prs)
    print(m_srs-std_srs, m_srs+std_srs)

    final_results[config] = [m_prs, std_prs, m_srs, std_srs]


with open(path_table + 'results_predictability_reachability_significative.txt', 'w') as f:

    for config in ['naive_literal', 'naive_pragmatic', 'pedagogical_literal', 'pedagogical_pragmatic']:

        f.write(config)
        f.write('\n')

        f.write('Predictability Mean:' + str(final_results[config][0]))
        f.write('\n')
        f.write('Predictability Std:' + str(final_results[config][1]))
        f.write('\n')
        f.write('Reachability Mean:' + str(final_results[config][2]))
        f.write('\n')
        f.write('Reachability Std:' + str(final_results[config][3]))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
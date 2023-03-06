import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import random


class JobEnv:
    def __init__(self, case_name, path, max_job=100, max_machine=20):
        self.case_name = case_name
        file = path + case_name + ".txt"
        with open(file, 'r') as f:
            user_line = f.readline()
            data = user_line.split('\t')
            self.m_n = list(map(int, data))
            data = f.read()
            data = str(data).replace('\n', '\t')
            data = str(data).split('\t')
            while data.__contains__(""):
                data.remove("")
            job = list(map(int, data))
            self.job = np.array(job).reshape(self.m_n[0], self.m_n[1] * 2)

        self.state_table = None
        self.job_num = self.m_n[0]
        self.machine_num = self.m_n[1]
        self.action_num = self.job_num
        self.max_job = self.job_num
        self.max_machine = self.machine_num
        self.current_time = 0  # current time
        self.finished_jobs = None

        self.next_time_on_machine = None
        self.job_on_machine = None
        self.current_op_of_job = None
        self.assignable_job = None

        self.init_op_of_job = np.repeat(0, self.job_num)  # init operation state of job
        self.init_finished_jobs = np.zeros(self.job_num, dtype=bool)
        self.init_assignable_jobs = np.ones(self.job_num, dtype=bool)
        for m in range(self.job_num):
            for n in range(self.machine_num):
                if self.job[m][n * 2] == -self.machine_num and self.job[m][n * 2 + 1] == 0:
                    self.init_op_of_job[m] += 1
                    if self.init_op_of_job[m] >= self.machine_num:
                        self.init_finished_jobs[m] = True
                        self.init_assignable_jobs[m] = False
                elif self.job[m][n * 2] < 0:
                    self.job[m][n * 2] = -self.job[m][n * 2]
                else:
                    break
        # state features are above 4 variables
        self.state_num = max_machine * 2 + max_job*2
        self.state = None

        self.max_op_len = 0
        self.total_process_time = 0
        # find maximum operation length of all jobs
        for j in range(self.job_num):
            job_len = 0
            for i in range(self.machine_num):
                self.total_process_time += self.job[j][i * 2 + 1]
                job_len += self.job[j][i * 2 + 1]
                if self.max_op_len < self.job[j][i * 2 + 1]:
                    self.max_op_len = self.job[j][i * 2 + 1]

        self.last_release_time = None
        self.done = False
        self.mask = None
        self.reward = 0
        self.no_op_cnt = 0

    def reset(self):
        self.current_time = 0  # current time
        self.next_time_on_machine = np.repeat(0, self.machine_num)
        self.job_on_machine = np.repeat(-1, self.machine_num)  # -1 implies idle machine
        self.current_op_of_job = np.repeat(0, self.job_num)  # current operation state of job
        self.assignable_job = np.ones(self.action_num, dtype=bool)  # whether a job is assignable
        self.finished_jobs = np.zeros(self.job_num, dtype=bool)

        self.finished_jobs[0:self.job_num] = self.init_finished_jobs
        self.current_op_of_job[0:self.job_num] = self.init_op_of_job
        self.assignable_job[0:self.job_num] = self.init_assignable_jobs

        self.last_release_time = np.repeat(0, self.job_num)
        self.state = np.zeros(self.state_num, dtype=float)
        self.assignable_job[self.job_num:self.action_num] = False  #
        self.mask = np.zeros(self.action_num, dtype=int)
        self.mask[self.job_num:self.action_num] = -1e8
        self.done = False
        self.no_op_cnt = 0
        self.state_table = copy.deepcopy(self.job)
        return self._get_state(), self.mask

    def _get_state(self):
        self.state[0:self.machine_num] = (self.next_time_on_machine - self.current_time) / self.max_op_len
        self.state[self.max_machine:self.max_machine + self.machine_num] = np.array(self.job_on_machine) / self.job_num
        self.state[self.max_machine * 2:self.max_machine * 2 + self.job_num] = self.current_op_of_job / self.machine_num
        self.state[self.max_machine * 2 + self.max_job: self.max_machine * 2 + self.max_job * 2] = self.assignable_job
        return self.state.flatten()

    def step(self, action):
        self.done = False
        self.reward = 0
        # action is operation
        self.allocate_job(action)
        if self.stop():
            self.done = True
        return self._get_state(), self.reward/self.max_op_len, self.done, self.mask

    def allocate_job(self, job_id):
        stage = self.current_op_of_job[job_id]
        machine_id = self.job[job_id][stage * 2]
        process_time = self.job[job_id][stage * 2 + 1]

        self.job_on_machine[machine_id] = job_id
        self.next_time_on_machine[machine_id] += process_time

        self.last_release_time[job_id] = self.current_time
        self.assignable_job[job_id] = False
        self.mask[job_id] = -1e8
        # assignable jobs whose current machine are employed will not be assignable
        for x in range(self.job_num):
            if self.assignable_job[x] and self.job[x][self.current_op_of_job[x] * 2] == machine_id:
                self.assignable_job[x] = False
                self.mask[x] = -1e8
                self.state_table[x][self.current_op_of_job[x] * 2] = -machine_id
        # there is no assignable jobs after assigned a job and time advance is needed
        # self.reward += process_time
        while sum(self.assignable_job) == 0 and not self.stop():
            self.reward -= self.time_advance()
            self.release_machine()

    def time_advance(self):
        hole_len = 0
        min_next_time = min(self.next_time_on_machine)
        if self.current_time < min_next_time:
            self.current_time = min_next_time
        else:
            self.current_time = self.find_second_min()
        for machine in range(self.machine_num):
            dist_need_to_advance = self.current_time - self.next_time_on_machine[machine]
            if dist_need_to_advance > 0:
                self.next_time_on_machine[machine] += dist_need_to_advance
                hole_len += dist_need_to_advance
            else:
                job = self.job_on_machine[machine]
                if self.current_op_of_job[job]<self.machine_num:
                    self.state_table[job][self.current_op_of_job[job]*2+1] = -dist_need_to_advance
        return hole_len

    def release_machine(self):
        for k in range(self.machine_num):
            cur_job_id = self.job_on_machine[k]
            if cur_job_id >= 0 and self.current_time >= self.next_time_on_machine[k]:
                self.job_on_machine[k] = -1
                self.last_release_time[cur_job_id] = self.current_time
                for x in range(self.job_num):  # release jobs on this machine
                    if not self.finished_jobs[x] and self.job[x][self.current_op_of_job[x] * 2] == k:
                        self.assignable_job[x] = True
                        self.mask[x] = 0
                        self.state_table[x][self.current_op_of_job[x] * 2] = k
                self.state_table[cur_job_id][self.current_op_of_job[cur_job_id] * 2] = -self.machine_num
                self.current_op_of_job[cur_job_id] += 1
                if self.current_op_of_job[cur_job_id] >= self.machine_num:
                    self.finished_jobs[cur_job_id] = True
                    self.assignable_job[cur_job_id] = False
                    self.mask[cur_job_id] = -1e8
                else:
                    next_machine = self.job[cur_job_id][self.current_op_of_job[cur_job_id] * 2]
                    if self.job_on_machine[next_machine] >= 0:  # 如果下一工序的机器被占用，则作业不可分配
                        self.assignable_job[cur_job_id] = False
                        self.mask[cur_job_id] = -1e8
                        self.state_table[cur_job_id][self.current_op_of_job[cur_job_id] * 2] = -next_machine

    def stop(self):
        if sum(self.current_op_of_job) < self.machine_num * self.job_num:
            return False
        return True

    def find_second_min(self):
        min_time = min(self.next_time_on_machine)
        second_min_value = 100000
        for value in self.next_time_on_machine:
            if min_time < value < second_min_value:
                second_min_value = value
        if second_min_value == 100000:
            return min_time
        return second_min_value

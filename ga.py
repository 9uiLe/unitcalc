#coding: utf-8

# ******************************
# Method
# ------------------------------
# Roulette Selection
# One point crossover
# Elite preservation

# 実行方法
# python ga.py

import random
import numpy as np
import os
import xlwings as xw
import matplotlib
import matplotlib.pyplot as plt

# ******************************
# Genetic Algorithm
# ------------------------------
class GA:
    # ******************************
    # Constructor
    # ------------------------------
    def __init__(self):
        # ******************************
        # Working Variable
        # ------------------------------
        self.POPULATION = 100# 一つの世代を構成する個体数
        self.GENE_MAX   = 1000# 何世代繰り返したらGAの処理を終了するか
        self.GENE_LEN   = 14#  受講する科目数
        self.CROS_RATE  = 95#  交叉率[%]
        self.MUTA_RATE  = 5#   突然変異率[%]
        self.ITEM_MAX   = 31#  科目数

        self.GENERAL    = 8  - 4#   一般科目の最低習得単位数
        self.SPECIALTY  = 44 - 30#  専門科目の最低習得単位数
        self.CREDIT     = 62 - 34#  卒業に必要な単位数

        self.retry      = False # 例外処理用

        self.subject_id      = [0]   * self.ITEM_MAX
        self.subject         = ["AAAAAAAAAAAAAAA"]   * self.ITEM_MAX
        self.credit          = np.zeros(self.ITEM_MAX,dtype = float)
        self.required        = ["AA"] * self.ITEM_MAX
        self.difficulity     = np.zeros(self.ITEM_MAX,dtype = float)
        self.attribute       = ["AA"] * self.ITEM_MAX

        self.population      = np.zeros([self.POPULATION,self.GENE_LEN],dtype = int)

        self.diff_total      = np.zeros(self.POPULATION,dtype = float)
        self.general_total   = np.zeros(self.POPULATION,dtype = float)
        self.specialty_total = np.zeros(self.POPULATION,dtype = float)
        self.credit_total    = np.zeros(self.POPULATION,dtype = float)


        self.elite        = [0] * self.GENE_LEN

        self.diff_graph   = [0.0] * self.GENE_MAX

        # ******************************
        # initialize
        # ------------------------------
        np.random.seed(0)

    # ******************************
    # Read CSV File
    # ------------------------------
    def ReadFile(self):
        self.wb = xw.Book("Book1.xlsm")
        sht = self.wb.sheets['Book1']
        for i in range(self.ITEM_MAX):
            self.subject_id[i]  = sht.range('A'+str(i+2)).value
            self.subject[i]     = sht.range('B'+str(i+2)).value
            self.credit[i]      = sht.range('C'+str(i+2)).value
            self.required[i]    = sht.range('D'+str(i+2)).value
            self.difficulity[i] = sht.range('E'+str(i+2)).value
            self.attribute[i]   = sht.range('F'+str(i+2)).value

    # ******************************
    # Make population
    # ------------------------------
    def MakePopulation(self):
        index_count = 0
        for i in range(self.POPULATION):
            arr = np.arange(0,self.ITEM_MAX)
            np.random.shuffle(arr)
            index = arr
            for j in range(self.GENE_LEN):
                self.population[i][j] = index[j]

    # ******************************
    # Calculation of sum of difficulity
    # ------------------------------
    def CalcSumDiff(self):
        for i in range(0,self.POPULATION):
            attr_count = 0
            spec_count = 0
            for j in range(0,self.GENE_LEN):
                self.diff_total[i]   += self.difficulity[ self.population[i][j] ]
                self.credit_total[i] += self.credit[ self.population[i][j] ]
                if self.attribute[j] == "一般":
                    self.general_total[i]   += self.credit[ self.population[i][j] ]
                elif self.attribute[j] == "専門":
                    self.specialty_total[i] += self.credit[ self.population[i][j] ]

    # ******************************
    # Search Elite
    # ------------------------------
    def SearchElite(self):
        elite_id  = 0
        diff_best = self.diff_total[0]
        for i in range(1,self.POPULATION):
            if diff_best > self.diff_total[i] and self.credit_total[i] >= self.CREDIT and self.general_total[i] >= self.GENERAL and self.specialty_total[i] >= self.SPECIALTY:
                elite_id  = i
                diff_best = self.diff_total[i]
        for i in range(0,self.GENE_LEN):
            self.elite[i] = self.population[elite_id][i]

    # ******************************
    # Roulette Selection
    # ------------------------------
    def RouletteSelection(self):
        total = sum(self.diff_total)
        temp_population = np.zeros([self.POPULATION,self.GENE_LEN],dtype=int)
        for i in range(0,self.POPULATION):
            rulet_target = np.random.uniform(0,total)
            rulet_val    = 0.0
            rulet_id     = 0

            for j in range(0,self.POPULATION):
                rulet_val += self.diff_total[j]
                if rulet_val > rulet_target:
                    rulet_id = j
                    break

            for j in range(0,self.GENE_LEN):
                temp_population[i][j] = self.population[rulet_id][j]

        for i in range(0,self.POPULATION):
            for j in range(0,self.GENE_LEN):
                self.population[i][j] = temp_population[i][j]

    # ******************************
    # Rank Selection
    # ------------------------------
    def RankSelection(self):
        temp_population = np.zeros([self.POPULATION,self.GENE_LEN],dtype=int)
        #RANK = np.arange(1,101)
        rank_id = 0
        RANK = np.arange(1,self.POPULATION+1)
        for k in range(self.POPULATION):
            ranking = [0] * self.POPULATION

            for j in range(0,self.POPULATION):
                ranking[j] = np.argsort(self.diff_total)[j]

            probability = np.random.rand()*5050
            rank_val = 0
            for i in range(0,self.POPULATION):
                rank_val += RANK[i]
                if rank_val > probability:
                    rank_id = i
                    break

            for j in range(0,self.GENE_LEN):
                temp_population[k][j] = self.population[rank_id][j]

        for i in range(0,self.POPULATION):
            for j in range(0,self.GENE_LEN):
                self.population[i][j] = temp_population[i][j]

    # ******************************
    # Crossover
    # ------------------------------
    def Crossover(self):
        for i in range(0,self.POPULATION,2):
            rate = np.random.uniform(0,100)
            if( rate < self.CROS_RATE ):

                # 一点交叉（改良版）
                pos  = np.random.randint(0,self.GENE_LEN)
                temp = 0
                for j in range(pos,self.GENE_LEN):
                    temp = self.population[i][j]
                    if self.population[i+1][j] not in self.population[i]:
                        self.population[i][j] = self.population[i+1][j]
                    if temp not in self.population[i+1]:
                        self.population[i+1][j] = temp

    # ******************************
    # Mutation
    # ------------------------------
    def Mutation(self):
        for i in range(0,self.POPULATION):
            for j in range(0,self.GENE_LEN):
                rate = np.random.uniform(0,100)
                if( rate < self.MUTA_RATE):
                    temp = np.random.randint(self.ITEM_MAX)
                    if temp not in self.population[i]:
                        self.population[i][j] = temp

    # ******************************
    # Save Elite
    # ------------------------------
    def SaveElite(self,gene_count):
        self.population[0] = self.elite
        for i in range(self.GENE_LEN):
            self.diff_graph[gene_count] += self.difficulity[self.elite[i]]

    # ******************************
    # Save Result
    # ------------------------------
    def SaveResult(self):
        # Book1.xlsmに'Result'シートを生成する
        if self.retry == False:
            try:
                # シートの追加をトライ
                xw.sheets.add(name="Result")
            except:
                # エラーが出たなら例外処理用変数をTrueにして再度実行
                self.retry = True
                return
        sht = self.wb.sheets['Result']
        sht.range('A1').value = "Subject_ID"
        sht.range('B1').value = "Subject"
        sht.range('C1').value = "Credit"
        sht.range('D1').value = "Required/Selected"
        sht.range('E1').value = "Attribute"
        sht.range('F1').value = "Difficulity"
        for i in range(self.GENE_LEN):
            sht.range('A' + str(i+2)).value = "{}".format(self.population[0][i]+1)
            sht.range('B' + str(i+2)).value = "{}".format(self.subject[self.population[0][i]])
            sht.range('C' + str(i+2)).value = "{}".format(self.credit[self.population[0][i]])
            sht.range('D' + str(i+2)).value = "{}".format(self.required[self.population[0][i]])
            sht.range('E' + str(i+2)).value = "{}".format(self.attribute[self.population[0][i]])
            sht.range('F' + str(i+2)).value = "{}".format(self.difficulity[self.population[0][i]])
        path = os.path.dirname(os.path.abspath(__file__))
        self.wb.save(path + "Book1.xlsm")

    # ******************************
    # Reset
    # ------------------------------
    def Reset(self):
        self.diff_total *= 0.0
        self.general_total *= 0.0
        self.specialty_total *= 0.0
        self.credit_total    *= 0.0

    # ******************************
    # Draw Difficulity Graph
    # ------------------------------
    def DrawDiffGraph(self):
        if self.retry == True:
            self.SaveResult()
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size']   = 16
        plt.figure()
        plt.title("Changes in Difficulity Level")
        plt.xlabel("Epoch")
        plt.ylabel("Total Difficulity")
        x = range(0,self.GENE_MAX)
        y = self.diff_graph
        plt.xlim([0,self.GENE_MAX])
        plt.plot(x,y,linewidth=1.5,color="red",label="difficulity")
        plt.savefig("difficulity.png",transparent=True,bbox_inches='tight',pad_inches=0.1,dpi=300)
        plt.close()

def main():
    ga = GA()
    ga.ReadFile()
    ga.MakePopulation()
    for i in range(ga.GENE_MAX):
        ga.CalcSumDiff()
        ga.SearchElite()
        ga.RouletteSelection()
        ga.Crossover()
        ga.Mutation()
        ga.SaveElite(i)
        ga.Reset()
    ga.SaveResult()
    ga.DrawDiffGraph()

if __name__ == "__main__":
    main()

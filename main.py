from ant_clustering import AntClustering

''' AntClustering params:
             grid:int       - Grid size                 defaults to 100
             rad:int        - Ant radius                defaults to 2
             antnum:int     - # agents                  defaults to 50
             iterations:int - # iterations              defaults to 5*10**6
             fname:string   - Dataset file              defaults to '400.txt'
             alpha:float    - Alpha value               defaults to 0
             sleep:bool     - Sleep 3s before starting  defaults to False
             dsize:int      - Display size              defaults to 500 '''

if __name__ == "__main__":
    antcluster = AntClustering(rad=2,
                               grid=100,
                               antnum=50,
                               fname='datasets/r15.txt',
                               alpha=1.6,
                               sleep=True)
    antcluster.run()

import daft
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

pgm = daft.PGM()
pgm.add_node("t", "time", 0, 0.75, aspect=1.5)
pgm.add_node("treat", "treatment", -0.75, 0, aspect=1.8)
pgm.add_node("y", "outcome", 0.75, 0, aspect=1.8)
pgm.add_edge("t", "y")
pgm.add_edge("t", "treat")
pgm.add_edge("treat", "y")
pgm.render()
pgm.savefig("DAG_interrupted_time_series.png", dpi=500)

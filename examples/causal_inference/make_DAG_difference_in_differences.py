import daft
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

pgm = daft.PGM()
pgm.add_node("treat", "treatment", 0, 0.5, aspect=1.8)
pgm.add_node("y", "outcome", 0, -0.5, aspect=1.5)
pgm.add_node("t", "time", -1, 0, aspect=1.2)
pgm.add_node("g", "group", 1, 0, aspect=1.2)
pgm.add_edge("t", "y")
pgm.add_edge("g", "y")
pgm.add_edge("t", "treat")
pgm.add_edge("g", "treat")
pgm.add_edge("treat", "y")
pgm.render()
pgm.savefig("DAG_difference_in_differences.png", dpi=500)

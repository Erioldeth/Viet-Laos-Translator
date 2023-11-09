from module.optim.adabelief import AdaBeliefOptim
from module.optim.adam import AdamOptim
from module.optim.scheduler import ScheduledOptim

optimizers = {
	"Adam": AdamOptim,
	"AdaBelief": AdaBeliefOptim
}

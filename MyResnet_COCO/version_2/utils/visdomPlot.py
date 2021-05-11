from visdom import Visdom
def plot_train(loss, acc, iteration, epoch, T):
    # apply visdom to plot train loss/acc-epoch curve
    viz = Visdom()
    x = T * epoch + iteration
    y_loss = loss
    y_acc = acc
    opt_loss = {
    'title' : ' Train loss',
    'xlabel': 'iteration',
    'ylabel': 'loss ',
    #'legend' : 'loss'
    }
    opt_acc = {
    'title' : ' Train  acc',
    'xlabel': 'iteration',
    'ylabel': 'acc',
    #'legend' : 'acc'
    }
    viz.line(X=[x], Y=[ y_acc], win='Train acc_iteration', update='append', opts=opt_acc)
    viz.line(X=[x], Y=[ y_loss], win='Train loss_iteration', update='append', opts=opt_loss)
    


def plot_val(loss, acc, epoch):
    # apply visdom to plot train loss/acc-epoch curve
    viz = Visdom()
    x = epoch
    y_loss = loss
    y_acc = acc
    opt_loss = {
    'title' : ' val loss',
    'xlabel': 'epoch',
    'ylabel': 'loss ',
    #'legend' : 'loss'
    }
    opt_acc = {
    'title' : ' val acc',
    'xlabel': 'epoch',
    'ylabel': 'acc',
    #'legend' :  'acc'
    }
    viz.line(X=[x], Y=[y_loss], win='Val loss', update='append', opts=opt_loss)
    viz.line(X=[x], Y=[y_acc], win='Val acc', update='append', opts=opt_acc)
import json
import matplotlib.pyplot as plt

metrics = []
iters =[]
total_loss = []
fast_rcnn_cls_accuracy = []
for line in open('metrics.json', 'r'):
    metrics.append(json.loads(line))
    if 'iteration' in metrics[-1] and 'total_loss' in metrics[-1] and \
            'fast_rcnn/cls_accuracy' in metrics[-1]:
        iters.append(metrics[-1]['iteration'])
        total_loss.append(metrics[-1]['total_loss'])
        fast_rcnn_cls_accuracy.append(metrics[-1]['fast_rcnn/cls_accuracy'])

# print(metrics[0])
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(iters, total_loss)
ax[0].set(xlabel='Iteration', ylabel='Total Loss', title='Training Curve')
ax[1].plot(iters, fast_rcnn_cls_accuracy)
ax[1].set(xlabel='Iteration', ylabel='Cls Accuracy', title='Training Curve')
plt.show()
print(metrics[0])
print(metrics[200])
b_dims = [[4, 5], [8, 10], [20, 25], [40, 50]]
c_dims = [[6, 6], [12, 12]]

train_percent = 0.6
val_percent = 0.2

fn_train = 'train_40x50.pkl'
fn_val = 'val_40x50.pkl'
fn_test = 'test_40x50.pkl'

parent_logger_name = "mrtl"

num_workers = 6
max_epochs = 50

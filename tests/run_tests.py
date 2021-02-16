import test_transforms
import test_color_transforms


def run_all(files):
    print('Running tests...\n')
    for f in files:
        for f_name in dir(f):
            if f_name.startswith('test'):
                print(f_name)
                func = getattr(f, f_name)
                if callable(func):
                    func()
        print()

    print('Passed all.')


if __name__ == '__main__':
    imported = [test_transforms, test_color_transforms]
    run_all(imported)


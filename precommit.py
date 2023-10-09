#! .venv/bin/python
"""
precommit
~~~~~~~~~

Things that should be done before committing changes to the repo.
"""
import configparser
import doctest
from fnmatch import fnmatch
import glob
import importlib
from itertools import zip_longest
import os
import sys
from textwrap import wrap

import mypy.api
import pycodestyle as pcs
import pytest
import rstcheck_core.checker as rstchecker


# Script configuration.
CONFIG_FILE = 'setup.cfg'


def get_config(filepath):
    """Pull configuration settings from the configuration file."""
    config_file = open(filepath)
    config = configparser.ConfigParser()
    config.read_file(config_file)
    return config['precommit']


def import_modules(names):
    """Import the modules needed for the doctest checks."""
    return [importlib.import_module(name) for name in names]


# Precommit checks.
def check_doctests(names):
    """Run documentation tests."""
    print('Running doctests...')
    if not names:
        print('No doctests found.')
    else:
        modules = import_modules(names)
        for mod in modules:
            doctest.testmod(mod)
        print('Doctests complete.')


def check_requirements():
    """Check requirements."""
    print('Checking requirements...')
    os.putenv('PIPENV_VERBOSITY', '-1')
    cmd = '.venv/bin/python -m pipenv requirements'
    current = os.popen(cmd).readlines()
    current = wrap_lines(current, 35, '', '  ')
    with open('requirements.txt') as fh:
        old = fh.readlines()
    old = wrap_lines(old, 35, '', '  ')

    # If the packages installed don't match the requirements, it's
    # likely the requirements need to be updated. Display the two
    # lists to the user, and let them make the decision whether
    # to freeze the new requirements.
    if current != old:
        print('requirements.txt out of date.')
        print()
        tmp = '{:<35} {:<35}'
        print(tmp.format('current', 'old'))
        print('\u2500' * 70)
        for c, o in zip_longest(current, old, fillvalue=''):
            print(tmp.format(c, o))
        print()
        update = input('Update? [y/N]: ')
        if update.casefold() == 'y':
            os.system(f'{cmd} > requirements.txt')
    os.unsetenv('PIPENV_VERBOSITY')
    print('Requirements checked...')


def check_rst(file_paths, ignore):
    """Remove trailing whitespace."""
    def action(files):
        results = []
        for file in files:
            with open(file) as fh:
                lines = fh.read()
            result = list(rstchecker.check_source(lines))
            if result:
                results.append(file, *result)
        return results

    def result_handler(result):
        if result:
            for line in result:
                print(' ' * 4 + line)

    title = 'Checking RSTs'
    file_ext = '.rst'
    run_check_on_files(title, action, file_paths, ignore,
                       file_ext, result_handler)


def check_style(file_paths, ignore):
    """Remove trailing whitespace."""
    def result_handler(result):
        if result.get_count():
            for msg in result.result_messages:
                lines = wrap(msg, 78)
                print(' ' * 4 + lines[0])
                for line in lines[1:]:
                    print(' ' * 6 + line)
            result.result_messages = []

    class StyleReport(pcs.BaseReport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.result_messages = []

        def error(self, line_number, offset, text, check):
            super().error(line_number, offset, text, check)
            msg = (f'{self.filename} {line_number}:{offset} {text}')
            self.result_messages.append(msg)

    title = 'Checking style'
    style = pcs.StyleGuide(config_file='setup.cfg', reporter=StyleReport)
    action = style.check_files
    file_ext = '.py'
    run_check_on_files(title, action, file_paths, ignore,
                       file_ext, result_handler)


def check_type_hints(path):
    """Check the type hinting."""
    print('Running type hinting check...')
    results = mypy.api.run([path, ])
    for report in results[:-1]:
        ps = report.split('\n')
        for p in ps:
            lines = wrap(p, initial_indent='  ', subsequent_indent='    ')
            for line in lines:
                print(line)
    print('Type hint checks complete.')


def check_unit_tests(path):
    """Run the unit tests."""
    print('Running unit tests...')
    result = pytest.main(['--capture', 'fd',])
    print('Unit tests complete.')
    return result


def check_venv():
    """Ensure this is running from the virtual environment for
    pjinoise. I know this is a little redundant with the shebang
    line at the top, but debugging issues caused by running from
    the wrong venv are a giant pain.
    """
    venv_path = '.venv/bin/python'
    dir_delim = '/'
    cwd = os.getcwd()
    exp_path = cwd + dir_delim + venv_path
    act_path = sys.executable
    if exp_path != act_path:
        msg = (f'precommit run from unexpected python: {act_path}. '
               f'Run from {exp_path} instead.')
        raise ValueError(msg)


def check_whitespace(file_paths, ignore):
    """Remove trailing whitespace."""
    title = 'Checking whitespace'
    action = remove_whitespace
    file_ext = '.py'
    run_check_on_files(title, action, file_paths, ignore, file_ext)


# Utility functions.
def get_module_dir():
    """Get the directory of the module."""
    cwd = os.getcwd()
    dirs = cwd.split('/')
    return f'{cwd}/src/{dirs[-1]}'


def in_ignore(name, ignore):
    for item in ignore:
        if fnmatch(name, item):
            return True
    return False


def run_check_on_files(title, action, file_paths, ignore,
                       file_ext=None, result_handler=None):
    print(f'{title}...')
    for file_path in file_paths:
        print(' ' * 2 + f'Checking {file_path}...', end='')
        files = glob.glob(file_path)
        if file_ext:
            files = [name for name in files if name.endswith(file_ext)]
        if ignore:
            files = [name for name in files if not in_ignore(name, ignore)]
        result = action(files)
        print('. Done.')
        if result and result_handler:
            result_handler(result)
    print(f'{title} complete.')
    return result


def remove_whitespace(filename):
    if isinstance(filename, (list, tuple)):
        for item in filename:
            remove_whitespace(item)
    else:
        with open(filename, 'r') as fh:
            lines = fh.readlines()
        newlines = [line.rstrip() for line in lines]
        newlines = [line + '\n' for line in newlines]
        with open(filename, 'w') as fh:
            fh.writelines(newlines)


def wrap_lines(lines, width, initial_indent, subsequent_indent):
    """Perform word wrapping on a sequence of lines of text."""
    out = []
    kwargs = {
        'width': width,
        'initial_indent': initial_indent,
        'subsequent_indent': subsequent_indent,
    }
    for line in lines:
        if line.endswith('\n'):
            line = line[:-1]
        wrapped = wrap(line, **kwargs)
        out.extend(wrapped)
    return out


def main():
    # Save time by not checking files that git ignores.
    ignore = []
    with open('./.gitignore') as fh:
        lines = fh.readlines()
    for line in lines:
        if line.endswith('\n'):
            line = line[:-1]
        if line:
            ignore.append(line)

    # Set up the configuration for the checks.
    config = get_config(CONFIG_FILE)
    doctest_modules = []
    if 'doctest_modules' in config:
        doctest_modules = config['doctest_modules'].split('\n')
    python_files = config['python_files'].split('\n')
    rst_files = config['rst_files'].split('\n')
    unit_tests = config['unit_tests']

    # Initial checks.
    check_venv()
    check_whitespace(python_files, ignore)
    result = check_unit_tests(unit_tests)

    # Only continue with precommit checks if the unit tests passed.
    if result == pytest.ExitCode.OK:
        check_requirements()
        check_doctests(doctest_modules)
        check_style(python_files, ignore)
        check_rst(rst_files, ignore)
        check_type_hints(get_module_dir())

    else:
        print('Unit tests failed. Precommit checks aborted. Do not commit.')


if __name__ == '__main__':
    main()

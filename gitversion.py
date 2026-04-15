import git


def get_version(lightweight=True):
    # Setting the git repository to current one
    repo = git.Repo(search_parent_directories=True)

    # Check if there are tags in the repository, returning 'untagged' if not
    if not repo.git.tag('--list'):
        # If repository is dirty [local changes], add '.local'
        if repo.is_dirty():
            return 'untagged.local'
        return 'untagged'

    # Adjusting parameters to include lightweight tags or not
    if lightweight:
        params = ['--tags', '--always']
    else:
        params = ['--always']

    # Get closest tag's name
    tag_name = repo.git.describe(*params+['--abbrev=0'])

    # Check if latest commit is the closest tag's commit
    is_same = repo.git.rev_list('-n', '1', f'{tag_name}') \
              == repo.git.rev_parse('HEAD')

    # If closest tag points to latest commit, distance is zero
    if is_same:
        version_name = f'{tag_name}.0'
    # Else, version is tag name with distance
    else:
        version_name = repo.git.describe(*params+['--abbrev=1'])
        version_name = '.'.join(version_name.split('-')[:-1])

    # If repository is dirty [local changes], add '.local' to version name
    if repo.is_dirty():
        return f'{version_name}.local'
    return version_name


if __name__ == '__main__':
    __version__ = get_version(lightweight=False)
    print(__version__)

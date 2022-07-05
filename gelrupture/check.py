import os
from git import Repo



def clean_git(filename):
    """Check whether the file belongs to a git repository in a clean state. If so, returns the current commit identifier."""
    script_path = os.path.dirname(os.path.abspath(filename))
    repo = Repo(script_path, search_parent_directories=True)
    assert not repo.bare
    assert not repo.is_dirty(), "You need to commit first your changes before running the analysis"
    return repo.head.commit

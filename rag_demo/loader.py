from typing import List
from langchain_community.document_loaders.markdown import UnstructuredFileLoader


class CustomMarkdownLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from rag_demo.partition_fenced import partition_md_fenced

        # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
        # versions of unstructured like 0.4.17-dev1
        _unstructured_version = __unstructured_version__.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )

        return partition_md_fenced(filename=self.file_path, **self.unstructured_kwargs)

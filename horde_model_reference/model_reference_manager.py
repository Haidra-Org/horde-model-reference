import json
from pathlib import Path

import horde_model_reference.path_consts as path_consts
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references
from horde_model_reference.legacy.download_live_legacy_dbs import LegacyReferenceDownloadManager
from horde_model_reference.model_reference_records import (
    MODEL_REFERENCE_TYPE_LOOKUP,
    CLIP_ModelReference,
    ControlNet_ModelReference,
    Generic_ModelReference,
    StableDiffusion_ModelReference,
)
from horde_model_reference.path_consts import MODEL_REFERENCE_CATEGORY


class ModelReferenceManager:
    """Class for downloading and reading model reference files."""

    _legacy_reference_download_manager: LegacyReferenceDownloadManager
    _cached_new_references: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}

    def __init__(
        self,
        download_and_convert_legacy_dbs: bool = True,
        override_existing: bool = True,
    ) -> None:
        """
        Initialize a new ModelReferenceManager instance.

        Args:
            download_and_convert_legacy_dbs: Whether to download and convert legacy model references.
            override_existing: Whether to override existing model reference files.
        """
        self._legacy_reference_download_manager = LegacyReferenceDownloadManager()
        if download_and_convert_legacy_dbs:
            self.download_and_convert_all_legacy_dbs(override_existing)

    def download_and_convert_all_legacy_dbs(self, override_existing: bool = True) -> bool:
        """
        Download and convert all legacy model reference files.

        Args:
            override_existing: Whether to override existing model reference files.
        """
        self._legacy_reference_download_manager.download_all_legacy_model_references(
            overwrite_existing=override_existing,
        )
        return convert_all_legacy_model_references()

    @property
    def all_legacy_model_reference_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """
        Get all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths.
        """
        return self.get_all_legacy_model_reference_file_paths(redownload_all=False)

    def get_all_legacy_model_reference_file_paths(
        self,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """
        Get all legacy model reference files.

        Args:
            redownload_all: Whether to redownload all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths.
        """
        return self._legacy_reference_download_manager.get_all_legacy_model_references(
            redownload_all=redownload_all,
        )

    @property
    def all_model_references(self) -> dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None]:
        """
        Get all model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths. Values of None indicate that the file does
            not exist (failed to download or convert).
        """
        return self.get_all_model_references(redownload_all=False)

    def get_all_model_references(
        self,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None]:
        """
        Get all model reference files.

        Args:
            redownload_all: Whether to redownload all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths. Values of None indicate that the file does
            not exist (failed to download or convert).
        """

        if not redownload_all and self._cached_new_references:
            return self._cached_new_references

        if redownload_all:
            self.download_and_convert_all_legacy_dbs()

        all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = path_consts.get_all_model_reference_file_paths()

        self._cached_new_references: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}

        for category, file_path in all_files.items():
            if file_path is None:
                self._cached_new_references[category] = None
            else:
                with open(file_path) as f:
                    file_contents = f.read()
                file_json: dict = json.loads(file_contents)

                parsed_model = MODEL_REFERENCE_TYPE_LOOKUP[category].model_validate(file_json)

                self._cached_new_references[category] = parsed_model

        return_dict: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}
        for reference_type, reference in self._cached_new_references.items():
            if reference is None:
                return_dict[reference_type] = None
                continue
            return_dict[reference_type] = reference.model_copy(deep=True)

        return return_dict

    @property
    def blip(self) -> Generic_ModelReference:
        """
        Get the BLIP model reference.

        Returns:
            The BLIP model reference.
        """
        blip = self.all_model_references[MODEL_REFERENCE_CATEGORY.blip]
        if blip is None:
            raise ValueError("BLIP model reference not found.")

        return blip

    @property
    def clip(self) -> CLIP_ModelReference:
        """
        Get the CLIP model reference.

        Returns:
            The CLIP model reference.
        """
        clip = self.all_model_references[MODEL_REFERENCE_CATEGORY.clip]
        if clip is None:
            raise ValueError("CLIP model reference not found.")

        if not isinstance(clip, CLIP_ModelReference):
            raise TypeError("CLIP model reference is not of the correct type.")

        return clip

    @property
    def codeformer(self) -> Generic_ModelReference:
        """
        Get the codeformer model reference.

        Returns:
            The codeformer model reference.
        """
        codeformer = self.all_model_references[MODEL_REFERENCE_CATEGORY.codeformer]
        if codeformer is None:
            raise ValueError("Codeformer model reference not found.")

        return codeformer

    @property
    def controlnet(self) -> ControlNet_ModelReference:
        """
        Get the controlnet model reference.

        Returns:
            The controlnet model reference.
        """
        controlnet = self.all_model_references[MODEL_REFERENCE_CATEGORY.controlnet]
        if controlnet is None:
            raise ValueError("ControlNet model reference not found.")

        if not isinstance(controlnet, ControlNet_ModelReference):
            raise TypeError("ControlNet model reference is not of the correct type.")

        return controlnet

    @property
    def esrgan(self) -> Generic_ModelReference:
        """
        Get the ESRGAN model reference.

        Returns:
            The ESRGAN model reference.
        """
        esrgan = self.all_model_references[MODEL_REFERENCE_CATEGORY.esrgan]
        if esrgan is None:
            raise ValueError("ESRGAN model reference not found.")

        return esrgan

    @property
    def gfpgan(self) -> Generic_ModelReference:
        """
        Get the GfPGAN model reference.

        Returns:
            The GfPGAN model reference.
        """
        gfpgan = self.all_model_references[MODEL_REFERENCE_CATEGORY.gfpgan]
        if gfpgan is None:
            raise ValueError("GfPGAN model reference not found.")

        return gfpgan

    @property
    def safety_checker(self) -> Generic_ModelReference:
        """
        Get the safety checker model reference.

        Returns:
            The safety checker model reference.
        """
        safety_checker = self.all_model_references[MODEL_REFERENCE_CATEGORY.safety_checker]
        if safety_checker is None:
            raise ValueError("Safety checker model reference not found.")

        return safety_checker

    @property
    def stable_diffusion(self) -> StableDiffusion_ModelReference:
        """
        Get the stable diffusion model reference.

        Returns:
            The stable diffusion model reference.
        """
        stable_diffusion = self.all_model_references[MODEL_REFERENCE_CATEGORY.stable_diffusion]

        if stable_diffusion is None:
            raise ValueError("Stable diffusion model reference not found.")

        if not isinstance(stable_diffusion, StableDiffusion_ModelReference):
            raise TypeError("Stable diffusion model reference is not of the correct type.")

        return stable_diffusion

    @property
    def miscellaneous(self) -> Generic_ModelReference:
        """
        Get the miscellaneous model reference.

        Returns:
            The miscellaneous model reference.
        """
        miscellaneous = self.all_model_references[MODEL_REFERENCE_CATEGORY.miscellaneous]
        if miscellaneous is None:
            raise ValueError("Miscellaneous model reference not found.")

        return miscellaneous

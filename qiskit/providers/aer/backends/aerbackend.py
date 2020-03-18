# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qiskit Aer qasm simulator backend.
"""

import copy
import json
import logging
import datetime
import os
import time
import uuid
import warnings
from numpy import ndarray

from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendStatus
from qiskit.qobj import validate_qobj_against_schema
from qiskit.result import Result

from qiskit.providers.aer.aerjob import AerJob
from qiskit.providers.aer.aererror import AerError


# Logger
logger = logging.getLogger(__name__)

# Location where we put external libraries that will be
# loaded at runtime by the simulator extension
LIBRARY_DIR = os.path.dirname(__file__)


class AerJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


class AerBackend(BaseBackend):
    """Qiskit Aer Backend class."""

    def __init__(self, configuration,
                 available_methods=None,
                 backend_options=None,
                 controller=None,
                 provider=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (BackendConfiguration): backend configuration.
            available_methods (list or None): the available simulation methods if
                                              backend is configurable.
            provider (BaseProvider): provider responsible for this backend.
            controller (function or None): Aer cython controller to be executed.
            backend_options (dict or None): set the default backend options.

        Raises:
            AerError: if there is no name in the configuration
        """
        # Initialize backend configuration.
        # The default configuration settings are stored in
        # `self._default_configuration`, and any custom configured options
        # are stored in `self._configuration`. The default values can be
        # restored from the default configuration using `reset_options`
        # method
        self._default_configuration = configuration
        super().__init__(copy.copy(self._default_configuration),
                         provider=provider)

        # Set Pybind11 C++ execution controller
        self._controller = controller

        # Set available methods
        if available_methods is None:
            self._available_methods = []
        else:
            self._available_methods = available_methods

        # Set custom configured options from backend_options dictionary
        self._options = {}
        if backend_options is not None:
            for key, val in backend_options.items():
                self._set_option(key, val)

    @property
    def options(self):
        """Return the current simulator options"""
        return self._options

    def set_options(self, **backend_options):
        """Set the simulator options"""
        for key, val in backend_options.items():
            self._set_option(key, val)

    def reset_options(self):
        """Reset the simulator options to default values."""
        self._configuration = self._default_configuration.copy()
        self._options = {}

    def available_methods(self):
        """Return the available simulation methods."""
        return self._available_methods

    # pylint: disable=arguments-differ
    def run(self, qobj,
            validate=True,
            backend_options=None,  # DEPRECATED
            **run_options):
        """Run a qobj on the backend.

        Args:
            qobj (QasmQobj): The Qobj to be executed.
            validate (bool): validate the Qobj before running (default: True).
            backend_options (dict or None): DEPRECATED dictionary of backend options
                                            for the execution (default: None).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            * kwarg options specified in ``run_options`` will override options
              of the same kwarg specified in the simulator options, the
              ``backend_options`` and the ``Qobj.config``.

            * The entries in the ``backend_options`` will be combined with
              the ``Qobj.config`` dictionary with the values of entries in
              ``backend_options`` taking precedence. This kwarg is deprecated
              and direct kwarg's should be used for options to pass them to
              ``run_options``.
        """
        # DEPRECATED
        if backend_options is not None:
            warnings.warn(
                'Using `backend_options` kwarg has been deprecated as of'
                ' qiskit-aer 0.5.0 and will be removed no earlier than 3'
                ' months from that release date. Runtime backend options'
                ' should now be added directly using kwargs for each option.',
                DeprecationWarning, stacklevel=3)
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, validate,
                         backend_options=backend_options, **run_options)
        aer_job.submit()
        return aer_job

    def run_direct(self, qobj, validate=True, **run_options):
        """Run a qobj on the backend.

        This executes a qobj and returns a Result object without initializing
        an async Job object.

        Args:
            qobj (QasmQobj): The Qobj to be executed.
            validate (bool): validate the Qobj before running (default: True).
            run_options (kwargs): additional run time backend options.

        Returns:
            Result: The simulation result.

        Additional Information:
            * kwarg options specified in ``run_options`` will override options
              of the same kwarg specified in the simulator options, the
              ``backend_options`` and the ``Qobj.config``.
        """
        # Submit job
        job_id = str(uuid.uuid4())
        return self._run_job(job_id, qobj, validate, **run_options)

    def status(self):
        """Return backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(backend_name=self.name(),
                             backend_version=self.configuration().backend_version,
                             operational=True,
                             pending_jobs=0,
                             status_msg='')

    def _run_job(self, job_id, qobj, validate,
                 backend_options=None,  # DEPRECATED
                 **run_options):
        """Run a job"""
        # Start timer
        start = time.time()
        run_config = self._run_config(
            backend_options=backend_options, **run_options)

        # Optional validation
        if validate:
            validate_qobj_against_schema(qobj)
            self._validate(qobj, run_config)

        # Run simulation
        output = self._execute(qobj, run_config)

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name())
            if output:
                logger.error('Output: %s', output)
            raise AerError("simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version

        # Add execution time
        output["time_taken"] = time.time() - start

        # Return results
        return Result.from_dict(output)

    def _set_option(self, key, value):
        """Special handling for setting backend options.

        This method should be extended by sub classes to
        update special option values.

        Args:
            key (str): key to update
            value (any): value to update.

        Raises:
            AerError: if key is 'method' and val isn't in available methods.
        """
        # If they key basis gates or coupling map we update the config
        if key == 'backend_name':
            self._configuration.backend_name = value
            return
        if key == 'basis_gates':
            self._configuration.basis_gates = value
            return
        if key == 'couling_map':
            self._configuration.coupling_map = value
            return

        # If key is method, we validate it is one of the available methods
        if key == 'method' and value not in self._available_methods:
            raise AerError("Invalid simulation method {}. Availalbe methods"
                           " are: {}".format(value, self._available_methods))
        # Add to options dict
        self._options[key] = value

    def _execute(self, qobj, run_config):
        """Run the controller"""
        # Add sim config to qobj
        controller_input = qobj.to_dict()
        for key, val in run_config.items():
            if hasattr(val, 'to_dict'):
                controller_input['config'][key] = val.to_dict()
            else:
                controller_input['config'][key] = val
        # Execute on controller
        return self._controller(controller_input)

    def _run_config(self,
                    backend_options=None,  # DEPRECATED
                    **run_options):
        """Return execution sim config dict from backend options."""
        # Get sim config
        run_config = self._options.copy()

        # Location where we put external libraries that will be
        # loaded at runtime by the simulator extension
        run_config['library_dir'] = LIBRARY_DIR

        # Override with run-time options
        if backend_options is not None:
            for key, val in backend_options.items():
                run_config[key] = val
        for key, val in run_options.items():
            run_config[key] = val
        return run_config

    def _validate(self, qobj, options):
        """Validate the qobj and backend_options for the backend"""
        pass

    def __repr__(self):
        """String representation of an AerBackend."""
        display = "backend_name='{}'".format(self.name())
        if self.provider():
            display += ', provider={}()'.format(self.provider())
        for key, val in self.options.items():
            display += ',\n    {}={}'.format(key, repr(val))
        return '{}(\n{})'.format(self.__class__.__name__, display)

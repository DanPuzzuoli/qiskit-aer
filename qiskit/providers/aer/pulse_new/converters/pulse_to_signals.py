import numpy as np
from typing import List

from qiskit.pulse import Schedule, Play, ShiftPhase, SetPhase, ShiftFrequency, SetFrequency
from qiskit import QiskitError
from qiskit.providers.aer.pulse_new.models.signals import PiecewiseConstant


class InstructionToSignals:
    """Converts pulse instructions to Signals for the Aer simulator."""

    def __init__(self, dt: int, carriers: List[float] = None):
        """

        Args:
            dt: length of the samples.
            carriers: a list of carrier frequencies. If it is not None there
                must be at least as many carrier frequencies as there are
                channels in the schedules that will be converted.
        """
        self._dt = dt
        self._carriers = carriers

    def get_signals(self, schedule: Schedule) -> List[PiecewiseConstant]:
        """
        Args:
            schedule: The schedule to represent in terms of signals.

        returns: a list of piecewise constant signals.
        """
        if self._carriers and len(self._carriers) < len(schedule.channels):
            raise QiskitError('Not enough carrier frequencies supplied.')

        signals, phases, frequency_shifts = {}, {}, {}

        for idx, ch in enumerate(schedule.channels):
            if self._carriers:
                carrier_freq = self._carriers[idx]
            else:
                carrier_freq = 0.

            phases[ch.name] = 0.
            frequency_shifts[ch.name] = 0.
            signals[ch.name] = PiecewiseConstant(samples=[], dt=self._dt, name=ch.name, carrier_freq=carrier_freq)

        for start_time, inst in schedule.instructions:
            ch = inst.channel.name
            phi = phases[ch]
            freq = frequency_shifts[ch]

            if isinstance(inst, Play):
                samples = []
                start_idx = len(signals[ch].samples)
                for idx, sample in enumerate(inst.pulse.get_waveform().samples):
                    t = self._dt * (idx + start_idx)
                    samples.append(sample * np.exp(1.0j * freq * t + 1.0j * phi))

                signals[ch].add_samples(start_time, samples)

            if isinstance(inst, ShiftPhase):
                phases[ch] += inst.phase

            if isinstance(inst, ShiftFrequency):
                frequency_shifts[ch] += inst.frequency

            if isinstance(inst, SetPhase):
                phases[ch] = inst.phase

            if isinstance(inst, SetFrequency):
                frequency_shifts[ch] = inst.frequency

        return list(signals.values())

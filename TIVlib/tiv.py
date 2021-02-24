import numpy as np
import matplotlib.pyplot as plt

epsilon = np.finfo(float).eps

class TIV:

    weights = [3, 8, 11.5, 15, 14.5, 7.5]
    shaath_profiles = [
        [0.0507767 + 0.0969534j, 0.776472 + 0.267636j, 0.948956 + 0.251339j, -0.16524 + -0.569966j, 1.53228 + 1.09356j,
         -0.0203655 + -8.91606e-06j],
        [0.0924506 + 0.0585745j, 0.620016 + -0.538637j, 0.25134 + -0.948967j, -0.410985 + 0.428071j,
         -0.780205 + -1.71321j, 0.0203654 + 1.19709e-06j],
        [0.109352 + 0.00450058j, -0.156464 + -0.806277j, -0.948969 + -0.251351j, 0.576213 + 0.141878j,
         -0.18093 + 1.87377j, -0.0203655 + -7.84055e-06j],
        [0.0969518 + -0.0507795j, -0.776489 + -0.267647j, -0.251354 + 0.948956j, -0.165235 + -0.569968j,
         1.09358 + -1.53229j, 0.0203655 + 1.60473e-08j],
        [0.058573 + -0.0924532j, -0.620034 + 0.538627j, 0.948955 + 0.251346j, -0.410989 + 0.428072j,
         -1.71321 + 0.78019j, -0.0203655 + -6.03267e-06j],
        [0.00449921 + -0.109355j, 0.156446 + 0.806264j, 0.251346 + -0.948972j, 0.576216 + 0.141871j, 1.87378 + 0.18091j,
         0.0203654 + -5.43203e-06j],
        [-0.0507812 + -0.0969548j, 0.776468 + 0.26764j, -0.948972 + -0.251354j, -0.165242 + -0.56996j,
         -1.53228 + -1.09358j, -0.0203654 + 1.37086e-06j],
        [-0.092455 + -0.058576j, 0.620017 + -0.538631j, -0.251357 + 0.948961j, -0.410979 + 0.428071j,
         0.780207 + 1.71319j, 0.0203654 + -9.56206e-06j],
        [-0.109356 + -0.00450247j, -0.156459 + -0.806279j, 0.948959 + 0.251343j, 0.57621 + 0.141868j,
         0.180922 + -1.87379j, -0.0203655 + -9.14619e-07j],
        [-0.0969567 + 0.0507781j, -0.776487 + -0.26765j, 0.251343 + -0.948966j, -0.165243 + -0.569953j,
         -1.09358 + 1.53228j, 0.0203654 + -3.65304e-06j],
        [-0.0585779 + 0.0924524j, -0.620036 + 0.538624j, -0.948967 + -0.251353j, -0.410973 + 0.42807j,
         1.71321 + -0.780213j, -0.0203655 + -2.18738e-06j],
        [-0.0045037 + 0.109354j, 0.156444 + 0.806271j, -0.251356 + 0.948958j, 0.576207 + 0.14187j, -1.87379 + -0.18094j,
         0.0203654 + -3.67884e-06j],
        [0.0802764 + -0.016992j, 0.592832 + -0.0724567j, 0.102923 + 0.797583j, 0.30574 + -0.318362j,
         1.54187 + -0.421998j, 0.134035 + -3.44708e-06j],
        [0.0610254 + -0.0548547j, 0.233667 + -0.549645j, 0.797582 + -0.102932j, -0.428579 + -0.105608j,
         -1.5463 + -0.405497j, -0.134035 + -3.81869e-06j],
        [0.0254222 + -0.0780199j, -0.359173 + -0.477196j, -0.102932 + -0.797596j, 0.122829 + 0.423949j,
         1.13638 + 1.12429j, 0.134035 + -4.78496e-06j],
        [-0.0169936 + -0.080279j, -0.592851 + 0.0724471j, -0.797597 + 0.102923j, 0.305737 + -0.318355j,
         -0.421983 + -1.54187j, -0.134035 + -2.05844e-07j],
        [-0.0548564 + -0.061028j, -0.233685 + 0.549639j, 0.102921 + 0.797588j, -0.428572 + -0.105609j,
         -0.405488 + 1.54628j, 0.134035 + -5.77259e-06j],
        [-0.078021 + -0.0254252j, 0.359158 + 0.477183j, 0.797587 + -0.102936j, 0.122826 + 0.423941j, 1.12431 + -1.1364j,
         -0.134035 + -5.36387e-06j],
        [-0.0802808 + 0.0169903j, 0.592832 + -0.0724572j, -0.102936 + -0.797597j, 0.305732 + -0.318351j,
         -1.54187 + 0.421975j, 0.134035 + -7.96623e-07j],
        [-0.0610301 + 0.0548533j, 0.233666 + -0.549645j, -0.797599 + 0.102924j, -0.428565 + -0.105608j,
         1.54629 + 0.405481j, -0.134035 + -6.56355e-06j],
        [-0.0254269 + 0.0780184j, -0.359173 + -0.477192j, 0.102922 + 0.797589j, 0.122823 + 0.423943j,
         -1.13638 + -1.12432j, 0.134035 + 2.79266e-07j],
        [0.0169888 + 0.0802778j, -0.592847 + 0.072444j, 0.797587 + -0.102935j, 0.305734 + -0.318354j,
         0.421976 + 1.54186j, -0.134035 + -9.74011e-06j],
        [0.0548515 + 0.0610271j, -0.233686 + 0.549633j, -0.102935 + -0.797599j, -0.428569 + -0.10561j,
         0.405495 + -1.54629j, 0.134035 + 2.91888e-06j],
        [0.0780164 + 0.0254243j, 0.359153 + 0.477187j, -0.7976 + 0.102925j, 0.122824 + 0.423949j, -1.12431 + 1.13637j,
         -0.134035 + -8.33809e-06j]]

    temperley_profiles = [
        [0.351315 + -0.829121j, -0.403439 + -2.2347j, -0.876702 + 0.432092j, -1.3074 + 1.75911j, 3.82005 + 5.50107j,
         0.829455 + -9.12661e-06j],
        [-0.149595 + -0.143645j, 1.19712 + -1.04144j, 0.585968 + -1.65992j, 0.103616 + 0.34265j, -0.756339 + -4.95562j,
         -0.0565238 + 8.59254e-06j],
        [-0.201376 + -0.0496023j, -0.303347 + -1.55746j, -1.65992 + -0.585972j, 0.244936 + -0.261059j,
         -1.82281 + 4.66986j, 0.0565238 + -9.32001e-06j],
        [-0.199198 + 0.0577297j, -1.50048 + -0.516036j, -0.585976 + 1.6599j, -0.348551 + -0.0816057j,
         3.91354 + -3.13283j, -0.0565238 + 8.7976e-07j],
        [-0.143645 + 0.149594j, -1.19714 + 1.04143j, 1.6599 + 0.585974j, 0.103604 + 0.342655j, -4.95563 + 0.756324j,
         0.0565237 + -3.23604e-06j],
        [-0.0496039 + 0.201372j, 0.303331 + 1.55745j, 0.585974 + -1.65992j, 0.244946 + -0.261077j, 4.66987 + 1.82278j,
         -0.0565239 + -1.15213e-05j],
        [0.0577278 + 0.199195j, 1.50045 + 0.516028j, -1.65992 + -0.58598j, -0.34857 + -0.0815952j, -3.13282 + -3.91352j,
         0.0565238 + 9.87602e-06j],
        [0.149591 + 0.143642j, 1.19712 + -1.04143j, -0.585985 + 1.65991j, 0.103622 + 0.342653j, 0.756333 + 4.95559j,
         -0.0565238 + -2.02276e-05j],
        [0.201371 + 0.0496003j, -0.303343 + -1.55747j, 1.65991 + 0.585966j, 0.244936 + -0.26109j, 1.82281 + -4.66987j,
         0.0565238 + 5.16088e-06j],
        [0.199192 + -0.0577307j, -1.50048 + -0.516037j, 0.585966 + -1.65991j, -0.348578 + -0.0815779j,
         -3.91354 + 3.13281j, -0.0565238 + -6.59373e-06j],
        [0.14364 + -0.149594j, -1.19714 + 1.04142j, -1.65991 + -0.585984j, 0.10364 + 0.342648j, 4.95563 + -0.756349j,
         0.0565238 + -2.77536e-06j],
        [0.0495993 + -0.201373j, 0.303324 + 1.55746j, -0.585986 + 1.65991j, 0.244925 + -0.26108j, -4.66987 + -1.82281j,
         -0.0565238 + 1.32486e-06j],
        [-0.00197941 + -0.195346j, 1.13406 + -1.75472j, 0.866344 + 2.07315j, 0.30886 + 1.08484j, 3.76853 + -0.815474j,
         -0.307674 + -1.43194e-05j],
        [-0.0993873 + -0.168186j, -0.952601 + -1.85949j, 2.07315 + -0.866352j, 0.785074 + -0.809913j,
         -3.67137 + -1.17807j, 0.307674 + 7.37818e-06j],
        [-0.170165 + -0.0959604j, -2.08667 + -0.104776j, -0.866352 + -2.07316j, -1.09394 + -0.274944j,
         2.59046 + 2.85592j, -0.307674 + -1.02584e-05j],
        [-0.195347 + 0.00197811j, -1.13408 + 1.75472j, -2.07316 + 0.866345j, 0.308863 + 1.08485j, -0.815431 + -3.76854j,
         0.307674 + 1.00458e-05j],
        [-0.168187 + 0.0993851j, 0.952591 + 1.85948j, 0.86634 + 2.07315j, 0.78508 + -0.809921j, -1.17809 + 3.67134j,
         -0.307674 + -1.978e-05j],
        [-0.0959614 + 0.170161j, 2.08666 + 0.104756j, 2.07315 + -0.866362j, -1.09395 + -0.274969j, 2.85593 + -2.59047j,
         0.307674 + 3.46413e-06j],
        [0.00197529 + 0.195344j, 1.13405 + -1.75472j, -0.86636 + -2.07316j, 0.308845 + 1.08487j, -3.76854 + 0.815436j,
         -0.307674 + -6.38597e-06j],
        [0.0993825 + 0.168183j, -0.952607 + -1.85949j, -2.07316 + 0.866343j, 0.785106 + -0.809919j, 3.67137 + 1.17806j,
         0.307674 + -2.46559e-06j],
        [0.17016 + 0.0959599j, -2.08667 + -0.104768j, 0.866339 + 2.07316j, -1.09396 + -0.274965j, -2.59046 + -2.85592j,
         -0.307674 + -4.55969e-08j],
        [0.195342 + -0.00197813j, -1.13407 + 1.75471j, 2.07316 + -0.866356j, 0.308853 + 1.08486j, 0.815438 + 3.7685j,
         0.307674 + -1.22027e-05j],
        [0.168182 + -0.0993859j, 0.952585 + 1.85947j, -0.866354 + -2.07317j, 0.785091 + -0.809921j, 1.17807 + -3.67136j,
         -0.307674 + 2.23251e-06j],
        [0.0959573 + -0.170162j, 2.08664 + 0.104772j, -2.07317 + 0.866348j, -1.09396 + -0.274952j, -2.85592 + 2.59045j,
         0.307674 + -3.2962e-06j]]

    key_labels = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'c', 'db', 'd', 'eb', 'e', 'f', 'gb',
                  'g', 'ab', 'a', 'bb', 'b']

    def __init__(self, energy, vector):
        self.energy = energy
        self.vector = vector

    def __add__(self, other):
        return self.combine(other)

    def __eq__(self, other):
        if np.array_equal(self.vector, other.vector):
            return True
        else:
            return False

    def __repr__(self):
        return "TIV object"

    def __str__(self):
        return self.vector, self.energy

    @classmethod
    def from_pcp(cls, pcp):
        fft = np.fft.rfft(pcp, n=12)
        energy = fft[0]
        vector = fft[1:7]
        if energy != 0:
            vector = ((vector / energy) * cls.weights)
        return cls(energy, vector)

    def phases(self):
        return np.angle(self.vector)

    def dissonance(self):
        return 1 - (np.linalg.norm(self.vector)/np.sqrt(np.sum(np.dot(self.weights, self.weights))))

    def combine(self, tiv2):
        return TIV(self.energy+tiv2.energy, (self.energy * self.vector + tiv2.energy * tiv2.vector) / (self.energy + tiv2.energy))

    def key(self, mode='temperley'):
        if mode == 'temperley':
            profiles = self.temperley_profiles
            alpha = 0.55
        else:
            profiles = self.shaath_profiles
            alpha = 0.2

        alpha_tiv = TIV(0, self.vector * alpha)
        distance = []

        for profile in profiles:
            distance.append(TIV.euclidean(alpha_tiv, TIV(0,profile)))

        index = np.argmin(distance)
        mode = 'maj'

        if index >= 12:
            mode = 'min'

        guessed_key = self.key_labels[index]
        return guessed_key, mode

    def mags(self):
        return np.abs(self.vector)

    def diatonicity(self):
        return self.mags()[4] / self.weights[4]

    def wholetoneness(self):
        return self.mags()[5] / self.weights[5]

    def chromaticity(self):
        return self.mags()[0] / self.weights[0]

    def plot_tiv(self, title=None):
        """
        Plot the TIV normalised vector inside circles
        :param title: Optional title for the plotted figure
        :return: None
        """
        titles = ["m2/M7", "TT", "M3/m6", "m3/M6", "P4/P5", "M2/m7"]
        tivs_vector = self.vector / self.weights
        i = 1
        for tiv in tivs_vector:
            circle = plt.Circle((0, 0), 1, fill=False)
            plt.subplot(2, 3, i)
            plt.subplots_adjust(hspace=0.4)
            plt.gca().add_patch(circle)
            plt.title(titles[i - 1])
            plt.scatter(tiv.real, tiv.imag)
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.grid()
            i = i + 1
        if title is not None:
            plt.gcf().suptitle(title)
        plt.show()


    def transpose(self, n_semitones, inplace=False):
        """
        Transpose the actual TIV by n semitones
        :param n_semitones: number of semitones to transpose (negative or positive)
        :param inplace: True to shift the actual TIV object. False to return a copy of the shifted TIV version
        :return: Shifted TIV version, or None
        """
        if n_semitones == 0:
            return self
        transpositions = self.get_12_transposes()
        transposition_desired = transpositions[n_semitones]
        if inplace:
            self.vector = transposition_desired.vector
            self.energy = transposition_desired.energy
        else:
            return transposition_desired


    def get_12_transposes(self):
        """
        Get all 12 possible transpositions of the vector
        :return: list containing the 12 transpositions
        """
        n = 12
        mod = np.abs(self.vector)
        phase = 1j * np.angle(self.vector)
        matmul = -2j * np.pi * (np.ones((6, 12), dtype=np.float64) * np.arange(12))
        semitones = np.arange(1, 7)
        semitones = semitones[:, np.newaxis]
        phase_transposition = semitones * matmul / n  # 12 phase transpositions for each interval
        transposed_phase = phase_transposition + phase[:, np.newaxis]
        transposed_vector = mod[:, np.newaxis] * np.exp(transposed_phase)
        return [TIV(self.energy, transposed_vector[:, i]) for i in range(12)]


    def small_scale_compatibility(self, cand_TIV):
        """
        Small scale compatibility between the actual TIV and the candidate TIV as defined in:
            Gilberto Bernardes, Diogo Cocharro, Marcelo Caetano, Carlos Guedes & Matthew E.P. Davies (2016)
            A multi-level tonal interval space for modelling pitch relatedness and musical consonance,
            Journal of New Music Research, 45:4, 281-294, DOI: 10.1080/09298215.2016.1182192
        :param cand_TIV: Candidate TIV object
        :return: The small scale compatibility
        """
        mixed_TIV = self.combine(cand_TIV)
        dissonance = mixed_TIV.dissonance()
        relatedness = TIV.euclidean(self, cand_TIV)
        dissonance_norm = 1 - (np.linalg.norm((self.vector + cand_TIV.vector)/2)/np.linalg.norm(self.weights))
        relatedness_norm = relatedness / (np.linalg.norm(self.weights)*2)
        return dissonance_norm * relatedness_norm

    def get_max_compatibility(self, tiv2):
        """
        Return the number of pitch shifts semitones that applied to tiv2 returns the maximum small scale compatibility.
        :param tiv2: The other tiv2 to compare to.
        :return: Number of pitch shifts to apply, small scale compatibility for that pitch shift.
        """
        tiv_tranpositions = tiv2.get_12_transposes()
        dissonances = []
        for tiv_tranposition in tiv_tranpositions:
            dissonances.append(self.small_scale_compatibility(tiv_tranposition))
        dissonances = np.array(dissonances)
        pitch_shift = np.argmin(dissonances)
        if pitch_shift > 5:
            pitch_shift = pitch_shift - 12
        return pitch_shift, min(dissonances)

    def hchange(self):
        tiv_array = self.vector
        results = []
        for i in range(len(tiv_array)):
            distance = TIV.euclidean(tiv_array[i + 1], tiv_array[i])
            results.append(distance)
        return results

    @classmethod
    def euclidean(cls, tiv1, tiv2):
        return np.linalg.norm(tiv1.vector - tiv2.vector)

    @classmethod
    def cosine(cls, tiv1, tiv2):
        tiv1_split = np.concatenate((tiv1.vector.real, tiv1.vector.imag), axis=0)
        tiv2_split = np.concatenate((tiv2.vector.real, tiv2.vector.imag), axis=0)
        return np.arccos(np.dot(tiv1_split, tiv2_split) / (np.linalg.norm(tiv1.vector) * np.linalg.norm(tiv2.vector)))


class TIVCollection(TIV):
    """
    Class to handle lists of TIV. To handle with ease compatibility between audio excerpts
    """
    def __init__(self, tivlist):
        """
        The constructor of the class. Takes a list of TIV vectors
        :param tivlist: A list containing all the tivs of an audio
        """
        if not all([isinstance(tivi, TIV) for tivi in tivlist]):
            raise TypeError("Some element in the list is not a TIV object")
        self.tivlist = tivlist
        self.energies = np.array([i.energy for i in tivlist])
        self.vectors = np.array([i.vector for i in tivlist])

    def __getitem__(self, item):
        return self.tivlist[item]

    def __repr__(self):
        return "TIVCollection (%s tivs)" % len(self.tivlist)

    def __str__(self):
        return self.tivlist

    @classmethod
    def from_pcp(cls, pcp):
        """
        Get TIVs from pcp, as the original method
        :param pcp: 12xN vector containing N pcps
        :return: TIVCollection object
        """
        if pcp.shape[0] != 12:
            raise TypeError("Vector is not compatible with PCP")
        fft = np.fft.rfft(pcp, n=12, axis=0)
        if fft.ndim == 1:
            fft = fft[:, np.newaxis]
        energy = fft[0, :] + epsilon
        vector = fft[1:7, :]
        vector = ((vector / energy) * np.array(cls.weights)[:, np.newaxis])
        return cls([TIV(energy[i], vector[:, i]) for i in range(len(energy))])

    def get_12_transposes(self):
        """
        Get all 12 possible transpositions for a TIVCollection
        :return:List with all 12 possible transpositions [0-11]
        """
        n = 12
        mod = np.abs(self.vectors)
        phase = 1j * np.angle(self.vectors)
        matmul = -2j * np.pi * (np.arange(12)[:, np.newaxis] * np.ones((6, 12, len(self.tivlist)), dtype=np.float64))
        matmul = np.transpose(matmul, axes=(2, 0, 1))
        semitones = np.arange(1, 7, dtype=np.float64)
        semitones = semitones[:, np.newaxis]
        phase_transposition = semitones * matmul / n
        new_phase = phase[:, :, np.newaxis] + phase_transposition
        new_vectors = mod[:, :, np.newaxis] * np.exp(new_phase)
        tivlists = []  # Will be length 12 containing all the 12 pitch shifts.
        for shift in range(n):
            set_tivs = []  # Aux variable to hold the set of tivs for this shift
            for tiv in range(len(self.energies)):
                set_tivs.append(TIV(self.energies[tiv], new_vectors[tiv, :, shift]))
            tivlists.append(TIVCollection(set_tivs))
        return tivlists

    def small_scale_compatibility(self, tivcol2):
        """
        Calculate small scale compatibility between two TIVCollections
        :param tivcol2: TIVCollections to compare
        :return: sum of the small scale harmonic compatibilities of all TIVs in collections
        """
        if len(tivcol2.tivlist) != len(self.tivlist):
            raise ValueError("Compatibility between different TIVCollections sizes are not supported yet")
        h_comps = np.zeros(len(tivcol2.tivlist))
        for idx in range(len(tivcol2.tivlist)):
            h_comps[idx] = self.tivlist[idx].small_scale_compatibility(tivcol2[idx])
        return np.sum(h_comps)

    def get_max_compatibility(self, tivcol2):
        """
        Get the pitch shift that minimizes the small scale compatibility measure
        :param tivcol2: TIVCollection object to compare against
        :return: A tuple containing pitch shift and small scale compatibility
        """
        tiv2transposes = tivcol2.get_12_transposes()
        compatibilities = np.zeros(12)
        for idx, transpose in enumerate(tiv2transposes):
            compatibilities[idx] = self.small_scale_compatibility(transpose)

        pitch_shift = np.argmin(compatibilities)
        if pitch_shift > 5:
            pitch_shift = pitch_shift - 12
        return pitch_shift, np.min(compatibilities)

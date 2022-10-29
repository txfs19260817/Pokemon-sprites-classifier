import { Data, Dex } from '@pkmn/dex';
import { Generations } from '@pkmn/data';

const NATDEX_UNOBTAINABLE_SPECIES = [
  'Eevee-Starter', 'Floette-Eternal', 'Pichu-Spiky-eared', 'Pikachu-Belle', 'Pikachu-Cosplay',
  'Pikachu-Libre', 'Pikachu-PhD', 'Pikachu-Pop-Star', 'Pikachu-Rock-Star', 'Pikachu-Starter',
  'Eternatus-Eternamax',
];

const NATDEX_EXISTS = (d: Data) => {
  // These checks remain unchanged from the default existence filter
  if (!d.exists) return false;
  if (d.kind === 'Ability' && d.id === 'noability') return false;
  // "National Dex" rules allows for data from the past, but not other forms of nonstandard-ness
  if ('isNonstandard' in d && d.isNonstandard) return false;
  // Unlike the check in the default existence function we don't want to filter out the 'Illegal' tier
  if ('tier' in d && d.tier === 'Unreleased') return false;
  // Filter out the unobtainable species
  if (d.kind === 'Species' && NATDEX_UNOBTAINABLE_SPECIES.includes(d.name)) return false;
  // Nonstandard items other than Z-Crystals and Pokémon-specific items should be filtered
  return !(d.kind === 'Item' && ['Past', 'Unobtainable'].includes(d.isNonstandard!) &&
    !d.zMove && !d.itemUser && !d.forcedForme);
};

const LEGALITY_FILTER = (d: Data) => {
    if (!d.exists) return false;
    if (d.kind === 'Ability' && d.id === 'noability') return false;
    if ('isNonstandard' in d && d.isNonstandard) return false;
    if (d.kind === 'Species' && NATDEX_UNOBTAINABLE_SPECIES.includes(d.name)) return false;
    return !(d.kind === 'Item' && ['Past', 'Unobtainable'].includes(d.isNonstandard!) &&
    !d.zMove && !d.itemUser && !d.forcedForme);
};

const gens = new Generations(Dex, NATDEX_EXISTS);

export const getDex = (gen: number) => gens.get(gen);
export const getLegalDex = () => new Generations(Dex, LEGALITY_FILTER).get(8);